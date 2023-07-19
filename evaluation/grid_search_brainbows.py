import argparse
import sys
from pathlib import Path
import importlib
from collections import defaultdict
import h5py
import numpy as np
import xarray as xr

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / 'cycleGAN'))

from evaluation.evaluate_brainbows import SEBrainbow, create_param_list
from evaluation.translate_image_old import translate_image
from evaluation.config.template import BrainbowSegmentationConfig, Config
from evaluation.evaluation_utils import get_path

def get_parameter_values(param):
    l = create_param_list(param)
    return [x[0] for x in l] # flatten list of 1-element tuples

def generate_evaluation(config_path, output_file):
    config: Config = importlib.import_module(f'config.{config_path}').config
    assert isinstance(config.segmentation_config, BrainbowSegmentationConfig)
    sconfig = config.segmentation_config
    tconfig = config.translate_image_config

    if sconfig.input_file is not None and Path(get_path(sconfig.input_file)).exists() and not tconfig.skip_translation:
        in_file, datasets = get_path(sconfig.input_file), sconfig.input_datasets
        assert datasets
        assert len(datasets) == 1
        with h5py.File(in_file) as f:
            images = [np.asarray(f[datasets[0]])]
    else:
        images = translate_image(tconfig)
        if tconfig.output_file is not None and tconfig.save_images:
            out_file, datasets = get_path(tconfig.output_file), tconfig.output_datasets
            assert datasets
            assert len(datasets) == 1
            with h5py.File(out_file, 'w-') as f:
                f.create_dataset(datasets[0], data=images[0])

    evaluater = SEBrainbow(config.segmentation_config)
    evaluater.find_segmentation_and_eval(images, True)

    n_bg_measures = len(create_param_list(evaluater.config.bg_measure))
    n_dist_measures = len(create_param_list(evaluater.config.dist_measure))
    n_bg_thresholds = len(create_param_list(evaluater.config.bg_threshold))
    n_bias_cuts = len(evaluater.bias_cut_values)

    coords = {
      'bg_threshold': (['bg_threshold'], get_parameter_values(evaluater.config.bg_threshold)),
      'bias_cut': (['bias_cut'], evaluater.bias_cut_values),
    }

    coord_names = ['bg_threshold']
    if evaluater.config.image_type == 'color':
        coord_names = ['bg_measure', 'dist_measure'] + coord_names
        normal_metric_shape = (n_bg_measures, n_dist_measures, n_bg_thresholds)
        coords = {
            'bg_measure': (['bg_measure'], get_parameter_values(evaluater.config.bg_measure)),
            'dist_measure': (['dist_measure'], get_parameter_values(evaluater.config.dist_measure)),
        } | coords
    else:
        normal_metric_shape = (n_bg_thresholds,)

    vi_coordinates = coord_names + ['bias_cut']
    vi_metric_shape = normal_metric_shape + (n_bias_cuts,)
    vi_metrics = ["under_seg", "over_seg", "VI",
                  "weighted_under_seg", "weighted_over_seg", "weighted_VI"]


    data = defaultdict(lambda: [])

    for evaluation in evaluater.results['evaluation']:
        assert len(evaluation['evaluation_scores']) == 1
        metrics = list(evaluation['evaluation_scores'].values())[0]
        for key, value in metrics.items():
            if type(value) == list:
                for searched_metrics in value:
                    for key2, value2 in searched_metrics.items():
                        data[key2].append(value2)
            elif key not in vi_metrics:
                data[key].append(value)

    data_vars = {}

    for key, value in data.items():
        if key in vi_metrics:
            data_vars[key] = (vi_coordinates, np.array(value).reshape(vi_metric_shape))
        else:
            data_vars[key] = (coord_names, np.array(value).reshape(normal_metric_shape))

    ds = xr.Dataset(data_vars, coords=coords, attrs=evaluater.results['config'])
    ds.attrs['offsets'] = np.array(ds.attrs['offsets']).flatten()
    ds.to_netcdf(output_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='The config file that will be used')
    parser.add_argument('output_file', type=str, help='The file where the results are saved to')

    args = parser.parse_args()
    generate_evaluation(args.config, args.output_file)
