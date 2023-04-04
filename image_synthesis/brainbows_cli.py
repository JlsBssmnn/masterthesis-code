import argparse
import importlib
import os
import pathlib
import sys
import h5py
sys.path.append(str(pathlib.Path(__file__).parent.parent.resolve()))
from image_synthesis.brainbows.colorize_brainbows import colorize_brainbows_cmap
from image_synthesis.logging_config import logging

def open_h5(path, *datasets_to_create):
    if os.path.isfile(path):
        f = h5py.File(path, 'a')
        action = 'appended to'
        if len(set(datasets_to_create).intersection(set(f.keys()))) != 0:
            logging.error(f'Some dataset you want to create already exists in {opt.output}')
            f.close()
            exit(1)
    else:
        action = 'created'
        f = h5py.File(opt.output, 'w-')
    return f, action

def main(opt):
    assert opt.output.endswith('.h5'), "Output file must be an h5 file"
    if not opt.config.startswith(opt.algorithm):
        logging.warning('Config %s seems to not match generation algorithm %s!', opt.config, opt.algorithm)

    label_dset_name = 'label'
    color_dset_name = 'color'
    if opt.name is not None:
        label_dset_name = opt.name + '_' + label_dset_name
        color_dset_name = opt.name + '_' + color_dset_name

    BrainbowGenerator = importlib.import_module('image_synthesis.brainbows.' + opt.algorithm).BrainbowGenerator
    config = importlib.import_module('config.brainbows.' + opt.config).config
    f, action = open_h5(opt.output, label_dset_name, color_dset_name)

    generator = BrainbowGenerator(config)
    image = generator.create_images()

    dataset = f.create_dataset(label_dset_name, data=image)
    attr = dataset.attrs
    attr['config'] = opt.config

    colorized = colorize_brainbows_cmap(opt.output, label_dset_name, config.cmap, config.cmap_dataset)
    dataset = f.create_dataset(color_dset_name, data=colorized)
    attr = dataset.attrs
    attr['config'] = opt.config
    attr['element_size_um'] = [0.2, 0.1, 0.1]

    logging.info(f'Successfully {action} file ${opt.output}')
    f.close()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('algorithm', type=str, help='The image generation algorithm that shall be used (located in ./brainbows).')
    ap.add_argument('config', type=str, help='The config file that stores the parameters for the image generation')
    ap.add_argument('output', type=str, help='The file where the output image should be stored (must be an h5 file)')
    ap.add_argument('-n', '--name', type=str, default=None, help='The base name of the datasets that store the output images')

    opt = ap.parse_args()
    main(opt)
