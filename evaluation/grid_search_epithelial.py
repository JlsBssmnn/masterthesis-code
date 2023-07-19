import argparse
import json
import sys
from pathlib import Path
from collections import deque
import datetime
import importlib
import re
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / 'cycleGAN'))


from evaluation.evaluate_epithelial import SEEpithelial
from evaluation.translate_image import translate_image
from cycleGAN.util.Evaluater import summarize_results
from utils.training_analysis.training_parser import parse_losses

seconds_in_day = 24 * 60 * 60

def get_epoch_length(file_path):
    f = open(file_path)

    def parse_date(s):
        match = re.search(r'\d\d/\d\d/\d\d \d\d:\d\d:\d\d', s)
        assert match is not None
        date = re.split(r'/| |:+', match.group())
        date = list(map(int, date))
        date[2] += 2000
        return datetime.datetime(date[2], date[1], date[0], date[3], date[4], date[5])

    def date_diff(date1, date2):
        diff = date2 - date1
        return diff.days * seconds_in_day + diff.seconds

    queue = deque()

    def add_line(line):
        nonlocal queue
        if len(queue) >= 2:
            queue.popleft()
        queue.append(line)

    epoch_lengths = []
    experiments = []
    experiment = start_date = None

    for line in f:
        if 'experiment' in line:
            if len(queue) == 2:
                end_date = parse_date(queue[0])
                epoch_count = re.search(r'End of epoch (\d+)', queue[0])
                assert epoch_count is not None
                epoch_count = int(epoch_count.group(1))

                experiments.append(experiment)
                epoch_lengths.append(date_diff(start_date, end_date) / epoch_count)


            experiment = re.search(r'experiment (\d+)', line)
            assert experiment is not None
            experiment = int(experiment.group(1))
            start_date = parse_date(line)
        add_line(line)
    else:
        end_date = parse_date(queue[0])
        epoch_count = re.search(r'End of epoch (\d+)', queue[0])
        assert epoch_count is not None
        epoch_count = int(epoch_count.group(1))

        experiments.append(experiment)
        epoch_lengths.append(date_diff(start_date, end_date) / epoch_count)


    f.close()
    df = pd.DataFrame({'epoch_length (s)': epoch_lengths}, experiments)
    df.sort_index(inplace=True)
    return df

def get_best_score_and_VI(results):
    """Uses a result dict produced by a SEEpithelial instance. It extracts the variation of information, over- and
    under segmentation and score for each slice when tweaking on that slice."""

    evaluation = results['evaluation']
    aggregate_metrics = [('variation_of_information', 'VI'), ('under_segmentation', 'under_seg'), ('over_segmentation', 'over_seg'), 'score']
    best_scores = {}

    for e in evaluation:
        seg_config = e['segmentation_parameters']
        tweak_image = seg_config['tweak_image']
        del seg_config['tweak_image']
        metrics = e['evaluation_scores'][tweak_image]

        for seg_param, value in seg_config.items():
            best_scores[tweak_image + '_' + seg_param] = value
        for metric in aggregate_metrics:
            if type(metric) == tuple:
                if metric[0] not in metrics:
                    print(f"Warning: Metric {metric[0]} isn't in the evaluation scores for {tweak_image}!")
                    continue
                best_scores[tweak_image + '_best_' + metric[1]] = metrics[metric[0]]
            else:
                if metric not in metrics:
                    print(f"Warning: Metric {metric} isn't in the evaluation scores for {tweak_image}!")
                    continue
                best_scores[tweak_image + '_best_' + metric] = metrics[metric]

    return pd.DataFrame(best_scores, index=[0])

def get_score_and_VI(grid_search_dir, config):
    config = importlib.import_module(f'config.{config}').config

    def set_generator_config(experiment_config, d):
        nonlocal config
        gen_config = config.translate_image_config.generator_config
        for key in list(vars(gen_config).keys()):
            if not key.startswith('__'):
                delattr(gen_config, key)

        gen_config.input_nc = experiment_config['input_nc']
        gen_config.output_nc = experiment_config['output_nc']
        gen_config.ngf = experiment_config['ngf']
        gen_config.netG = experiment_config['netG']
        gen_config.norm = experiment_config['norm']
        gen_config.use_dropout = not experiment_config['no_dropout']
        gen_config.init_type = experiment_config['init_type']
        gen_config.init_gain = experiment_config['init_gain']
        gen_config.gpu_ids = experiment_config['gpu_ids']

        for key, value in experiment_config['generator_config'].items():
            setattr(gen_config, key, value)

        config.translate_image_config.generator_save = str(d / 'latest_net_G_A.pth')
        config.segmentation_config.save_file_name = f"experiment_{d.name}"
        config.segmentation_config.membrane_black = experiment_config['evaluation_config']['membrane_black']

        ranges = ['membrane_range', 'basins_range']
        if experiment_config['evaluation_config']['membrane_black']:
            ranges.reverse()

        start, end, step = getattr(config.segmentation_config, ranges[0])
        setattr(config.segmentation_config, ranges[0], (start + (256-end), 256, step))
        start, end, step = getattr(config.segmentation_config, ranges[1])
        setattr(config.segmentation_config, ranges[1], (0, end-start, step))


    directories = Path(grid_search_dir)
    df = pd.DataFrame()

    for d in directories.iterdir():
        if not d.name.isdigit() or not d.is_dir():
            continue

        experiment = int(d.name)

        with open(d / 'config.json') as f:
            experiment_config = json.load(f)

        set_generator_config(experiment_config, d)
        images = translate_image(config.translate_image_config)
        evaluater = SEEpithelial(config.segmentation_config)
        evaluater.eval_and_store(images)

        results = summarize_results(evaluater.results, config.segmentation_config.image_names,
            [('variation_of_information', 'VI'), ('under_segmentation', 'under_seg'), ('over_segmentation', 'over_seg'), 'score'],
            ['diff'])
        df = pd.concat((df, pd.DataFrame(results, index=[experiment])))
    return df

def extract_best_score_and_VI(eval_dir, output_csv):
    """Takes a directory that contains subdirectories with results from a SEEpithelial instance. It extracts the
    variation of information, over- and under segmentation and score for each slice when tweaking on it and summarizes
    this information for all experiments in the directory into a csv."""
    df = pd.DataFrame()

    for path in Path(eval_dir).iterdir():
        if not path.name.startswith('experiment_'):
            continue
        try:
            experiment = int(path.stem[11:])
        except:
            continue

        with open(path) as f:
            results = json.load(f)
        best_scores = get_best_score_and_VI(results)
        index = best_scores.index.tolist()
        index[0] = experiment
        best_scores.index = index
        df = pd.concat((df, best_scores), axis=0)
    df.sort_index(inplace=True)
    df.to_csv(output_csv)

def get_min_diff(grid_search_dir):
    directories = Path(grid_search_dir)
    columns = ['slice1_diff', 'slice2_diff', 'slice3_diff']
    mins = np.empty((0, len(columns)))
    arg_mins = np.empty((0, len(columns)), dtype=int)
    index = []

    for d in directories.iterdir():
        if not d.name.isdigit() or not d.is_dir():
            continue
        index.append(int(d.name))
        losses = parse_losses(d / 'loss_log.txt')[columns]
        mins = np.append(mins, losses.min().values[None], axis=0)
        arg_mins = np.append(arg_mins, losses.idxmin().values[None], axis=0)

    df_columns = ['slice1_min_diff', 'slice2_min_diff', 'slice3_min_diff', 'slice1_argmin_diff', 'slice2_argmin_diff', 'slice3_argmin_diff']
    data = [mins, arg_mins]
    df = pd.DataFrame({df_columns[i]: data[i > 2][:, i % 3] for i in range(len(df_columns))}, index=index)
    df.sort_index(inplace=True)

    return df

def generate_evaluation(log_file, config, output_csv):
    result_dir = Path(log_file).parent

    duration_df = get_epoch_length(log_file)
    score_df = get_score_and_VI(result_dir, config)
    min_diff_df = get_min_diff(result_dir)

    df = pd.concat((duration_df, score_df, min_diff_df), axis=1)
    overfitting_scores = df[['slice1_diff', 'slice2_diff', 'slice3_diff']].values - \
        df[['slice1_min_diff', 'slice2_min_diff', 'slice3_min_diff']].values
    df = pd.concat((df, pd.DataFrame(overfitting_scores,
                    columns=['slice1_overfitting', 'slice2_overfitting', 'slice3_overfitting'])), axis=1)

    df.to_csv(output_csv)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('function', type=str, help='Which function in this file shall be run')
    parser.add_argument('function_params', nargs='*', help='The arguemnts that are passed to the function')

    args = parser.parse_args()

    globals()[args.function](*args.function_params)
