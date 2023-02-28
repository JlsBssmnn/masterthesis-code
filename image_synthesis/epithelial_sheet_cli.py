import argparse
import importlib
import os
import sys
import h5py
sys.path.append('..')
from image_synthesis.epithelial_sheet.v1_0 import generate_image

def main(config_path: str, output_path: str, dataset_name: str):
    assert output_path.endswith('.h5'), "Output file must be an h5 file"

    if dataset_name is None:
        dataset_name = f'epithelial_{config_path}'

    if os.path.isfile(output_path):
        f = h5py.File(output_path, 'a')
        action = 'appended to'
        if dataset_name in f:
            print(f'The given dataset name already exists in {output_path}')
            f.close()
            exit(1)
    else:
        action = 'created'
        f = h5py.File(output_path, 'w-')
    
    config = importlib.import_module('config.epithelial_sheet.' + config_path).config
    image = generate_image(config)

    dataset = f.create_dataset(dataset_name, data=image.T)
    attr = dataset.attrs
    attr['element_size_um'] = [1, 0.20024092, 0.20024027]
    attr['config'] = config_path

    print(f'Successfully {action} file ${output_path}')
    f.close()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('config', type=str, help='The config file that stores the parameters for the image generation')
    ap.add_argument('output', type=str, help='The file where the output image should be stored (must be an h5 file)')
    ap.add_argument('-n', '--name', type=str, default=None, help='The name of the dataset that stores the output image')

    args = ap.parse_args()
    main(args.config, args.output, args.name)
