import argparse
from pathlib import Path
from skimage import io

import h5py

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help='The file where the slice shall be extracted from')
    parser.add_argument('output_file', help='The file where to store the sliced image')
    parser.add_argument('--input_dataset', required=True, help='The dataset that contains the image')
    parser.add_argument('--output_dataset', default="slice", help='The dataset name where the slice is saved to')
    parser.add_argument('--slice', required=True, help='A string that is used to slice the image. This should be python code')
    parser.add_argument('--omit_slice_string', action='store_true', help='If set, the slice string is not stored on the h5 dataset')

    opt = parser.parse_args()
    in_suff = Path(opt.input_file).suffix
    out_suff = Path(opt.output_file).suffix

    if in_suff == '.h5':
        with h5py.File(opt.input_file) as f:
            image = eval(f'f["{opt.input_dataset}"][{opt.slice}]')
    else:
        image = io.imread(opt.input_file)
        image = eval(f"image[{opt.slice}]")

    if out_suff == '.h5':
        if Path(opt.output_file).is_file():
            with h5py.File(opt.output_file, 'a') as f:
                if opt.output_dataset in f:
                    raise ValueError(f'The dataset {opt.output_dataset} already exists!')
                d = f.create_dataset(opt.output_dataset, data=image)
                if not opt.omit_slice_string:
                    d.attrs['slice'] = opt.slice
        else:
            with h5py.File(opt.output_file, 'w-') as f:
                d = f.create_dataset(opt.output_dataset, data=image)
                if not opt.omit_slice_string:
                    d.attrs['slice'] = opt.slice
    else:
        io.imsave(opt.output_file, image)
