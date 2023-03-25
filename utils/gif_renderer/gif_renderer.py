"""
A utility for converting 3D images to gif files that display 2d slices of the images.
"""

import argparse

import os
from os import path
import h5py
from typing import cast
import numpy as np
from PIL import Image
import sys
import importlib

def extract_datasets(file, directory):
    if type(file[directory]) == h5py.Dataset:
        datasets = [file[directory]]
    elif type(file[directory]) == h5py.Group:
        datasets = []
        group = cast(h5py.Group, file[directory])
        for child in group:
            if type(group[child]) == h5py.Dataset:
                datasets.append(group[child])
    else:
        raise Exception("The provided directory is neither dataset nor group")
    return datasets

def main(opt):
    f = h5py.File(opt.input_file)
    directories = opt.directories
    os.makedirs(opt.output_dir, exist_ok=True)
    images = []

    for directory in directories:
        images.extend(extract_datasets(f, directory))

    def index(i: int, dim: int):
        return tuple([(slice(None) if x != opt.slice_axis else i) for x in range(dim)])

    def get_image_name(full_dataset_name, image_index):
        if image_index < len(opt.names):
            name = opt.names[image_index]
            if not name.endswith('.gif'):
                name += '.gif'
            return name
        else:
            dataset_name = full_dataset_name[full_dataset_name.rfind('/') + 1:]
            if dataset_name == '':
                return 'output.gif'
            else:
                return dataset_name + '.gif'

    for i, dataset in enumerate(images):
        image = np.asarray(dataset)
        if image.ndim == 4:
            assert np.argmin(image.shape) == 0, "Color channel must be in the first dimension"
            if image.shape[0] == 1:
                image = image[0]
            else:
                image = np.moveaxis(image, 0, -1)
        else:
            assert image.ndim == 3, "Images must be either 3D or 4D"

        dim = image.ndim
        initial_frame = Image.fromarray(image[index(0, dim)])
        additional_frames = [Image.fromarray(image[index(x, dim)]) for x in range(1, image.shape[opt.slice_axis])]
        if opt.add_reverse:
            additional_frames += list(reversed(additional_frames))[1:-1]

        initial_frame.save(path.join(opt.output_dir, get_image_name(dataset.name, i)),
                           format='GIF', append_images=additional_frames, save_all=True,
                           duration=opt.duration, loop=opt.loop)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Not enough arguments.')
        exit()
    elif sys.argv[1] == 'config':
        opt = importlib.import_module(sys.argv[2]).options
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('input_file', type=str, help='The file that contains the 3D image')
        parser.add_argument('output_dir', type=str, help='The directory where the gif files are stored')
        parser.add_argument('-d', '--directories', default='/', nargs='+', type=str, help='The dataset or group in the h5 file that contains the 3D images')
        parser.add_argument('-n', '--names', default=[], nargs='+', type=str, help='The names of the output files. If less names than files are provided, \
                            the default naming is applied after the name list is depleted which may overwrite created gifs')
        parser.add_argument('-s', '--slice_axis', default=0, type=int, help='The axis in the image that the gif is moving along over the frames')
        parser.add_argument('--duration', default=100, type=int, help='The duration of one gif frame in milliseconds')
        parser.add_argument('--loop', default=0, type=int, help='How often the gif loops, 0 means infinitelly')
        parser.add_argument('--add_reverse', action='store_true', help='If set, the reversed order of frames is added to the gif to enable smooth loops')

        opt = parser.parse_args()

    main(opt)
