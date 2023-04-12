import argparse
import importlib
import os
import sys
import h5py
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.resolve()))
from image_synthesis.logging_config import logging

def main(opt):
    assert opt.output.endswith('.h5'), "Output file must be an h5 file"
    if not opt.config.startswith(opt.algorithm):
        logging.warning('Config %s seems to not match generation algorithm %s!', opt.config, opt.algorithm)

    if opt.name is None:
        opt.name = f'epithelial_{opt.config}'

    if os.path.isfile(opt.output):
        f = h5py.File(opt.output, 'a')
        action = 'appended to'
        if opt.name in f:
            print(f'The given dataset name already exists in {opt.output}')
            f.close()
            exit(1)
    else:
        action = 'created'
        f = h5py.File(opt.output, 'w-')
    
    config = importlib.import_module('config.epithelial_sheet.' + opt.config).config
    algorithm = importlib.import_module('image_synthesis.epithelial_sheet.' + opt.algorithm).generate_image
    image = algorithm(config)

    dataset = f.create_dataset(opt.name, data=image[None, :])
    attr = dataset.attrs
    attr['element_size_um'] = [0.5, 0.2, 0.2]
    attr['config'] = opt.config

    print(f'Successfully {action} file ${opt.output}')
    f.close()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('algorithm', type=str, help='The image generation algorithm that shall be used (located in ./epithelial_sheet).')
    ap.add_argument('config', type=str, help='The config file that stores the parameters for the image generation')
    ap.add_argument('output', type=str, help='The file where the output image should be stored (must be an h5 file)')
    ap.add_argument('-n', '--name', type=str, default=None, help='The name of the dataset that stores the output image')

    opt = ap.parse_args()
    main(opt)
