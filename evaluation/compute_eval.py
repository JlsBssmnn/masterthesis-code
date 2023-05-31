import json

import h5py
import numpy as np
from pathlib import Path
from skimage.metrics import adapted_rand_error, variation_of_information
from partition_comparison import rand_index
from evaluation.config.template import Config
from evaluation.evaluation_utils import get_path

def determine_file_path(config: Config):
    """
    Looks in the output directory and determines an appropriate name for the output.
    """
    assert config.evaluation_config.save_directory is not None
    path = Path(get_path(config.evaluation_config.save_directory))
    assert path.is_dir()
    
    name = Path(get_path(config.translate_image_config.generator_save)).stem
    duplicates = list(path.glob(name + '*'))

    if len(duplicates) == 0:
        return path / (name + '.json')

    version = 1
    for duplicate in duplicates:
        last_part = duplicate.stem.split('_')[-1]
        if not last_part.isdigit():
            continue
        version = max(int(last_part) + 1, version)
    file_name = f'{name}_{version}.json'

    return path / file_name

def compute_eval(config: Config, images, segmentation_config):
    assert len(images) == len(config.evaluation_config.ground_truth_datasets)
    ground_truths = []
    datasets = config.evaluation_config.ground_truth_datasets

    with h5py.File(get_path(config.evaluation_config.ground_truth_file), 'r') as f:
        for dataset_list in datasets:
            image_ground_truths = []
            for dataset in dataset_list:
                image_ground_truths.append(np.asarray(f[dataset]))
            ground_truths.append(image_ground_truths)

    result = {'segmentation_config': segmentation_config, 'evaluation_scores': {}}

    for i, image in enumerate(images):
        for dataset, ground_truth in zip(datasets[i], ground_truths[i]):
            over_segmentation, under_segmentation = variation_of_information(ground_truth, image)
            voi = over_segmentation + under_segmentation
            rand_index_value = rand_index(ground_truth.flatten(), image.flatten())
            are, prec, rec = adapted_rand_error(ground_truth, image)

            evaluation = {
                'Over segmentation': over_segmentation,
                'Under segmentation': under_segmentation,
                'Variation of information': voi,
                'Rand index': rand_index_value,
                'adapted Rand error (alpha=0.5)': are,
                'adapted Rand precision': prec,
                'adapted Rand recall': rec,
            }

            result['evaluation_scores'][dataset] = evaluation

    with open(determine_file_path(config), 'x') as f:
        f.write(json.dumps(result, indent=4))


def one_step(config: Config):
    eval_config = config.evaluation_config
    assert eval_config.segmentation_file is not None
    assert eval_config.config_file is not None
    assert eval_config.segmentation_datasets is not None and \
        len(eval_config.segmentation_datasets) == len(eval_config.ground_truth_datasets)
    
    with open(get_path(eval_config.config_file), 'r') as f:
        segmentation_config = json.load(f)

    images = []
    with h5py.File(get_path(eval_config.segmentation_file)) as f:
        for dataset in eval_config.segmentation_datasets:
            images.append(np.asarray(f[dataset]))

    compute_eval(config, images, segmentation_config)
