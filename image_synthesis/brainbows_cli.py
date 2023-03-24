import argparse
import importlib
import os
import pathlib
import sys
import h5py
sys.path.append(str(pathlib.Path(__file__).parent.parent.resolve()))
from image_synthesis.brainbows.colorize_brainbows import colorize_brainbows_cmap
from image_synthesis.brainbows.v_1_1 import BrainbowGenerator
from image_synthesis.logging_config import logging

def main(config_path: str, output_path: str, dataset_name: str):
    assert output_path.endswith('.h5'), "Output file must be an h5 file"

    label_dset_name = 'label'
    color_dset_name = 'color'
    if dataset_name is not None:
        label_dset_name = dataset_name + '_' + label_dset_name
        color_dset_name = dataset_name + '_' + color_dset_name

    if os.path.isfile(output_path):
        f = h5py.File(output_path, 'a')
        action = 'appended to'
        if label_dset_name in f or color_dset_name in f:
            logging.error(f'The given dataset name already exists in {output_path}')
            f.close()
            exit(1)
    else:
        action = 'created'
        f = h5py.File(output_path, 'w-')
    
    config = importlib.import_module('config.brainbows.' + config_path).config
    generator = BrainbowGenerator(config)
    image = generator.create_images()

    dataset = f.create_dataset(label_dset_name, data=image)
    attr = dataset.attrs
    attr['config'] = config_path

    colorized = colorize_brainbows_cmap(output_path, label_dset_name, config.cmap, config.cmap_dataset)
    dataset = f.create_dataset(color_dset_name, data=colorized)
    attr = dataset.attrs
    attr['config'] = config_path
    attr['element_size_um'] = [0.2, 0.1, 0.1]

    logging.info(f'Successfully {action} file ${output_path}')
    f.close()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('config', type=str, help='The config file that stores the parameters for the image generation')
    ap.add_argument('output', type=str, help='The file where the output image should be stored (must be an h5 file)')
    ap.add_argument('-n', '--name', type=str, default=None, help='The base name of the datasets that store the output images')

    args = ap.parse_args()
    main(args.config, args.output, args.name)
