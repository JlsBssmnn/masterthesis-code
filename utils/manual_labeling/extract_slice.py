import argparse

import h5py


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help='The file where the slice shall be extracted from')
    parser.add_argument('output_file', help='The file where to store the sliced image')
    parser.add_argument('--dataset', required=True, help='The dataset that contains the image')
    parser.add_argument('--slice', required=True, help='A string that is used to slice the image. This should be python code')

    opt = parser.parse_args()

    with h5py.File(opt.input_file) as f:
        image = eval(f'f["{opt.dataset}"][:, {opt.slice}]')

    with h5py.File(opt.output_file, 'w-') as f:
        f.create_dataset('slice', data=image)
