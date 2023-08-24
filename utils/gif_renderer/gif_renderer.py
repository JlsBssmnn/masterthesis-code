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
from skimage.color import label2rgb
import importlib
import filters

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

def create_filters(opt):
    filter_functions = [getattr(filters, filter_name) for filter_name in opt.filters]
    def composed_filter(image):
        for filter_name, filter in zip(opt.filters, filter_functions):
            image = filter(image, **opt.filters[filter_name])
        return image

    return composed_filter

def main(opt):
    f = h5py.File(opt.input_file)
    directories = opt.directories
    os.makedirs(opt.output_dir, exist_ok=True)
    images = []
    filter_func = create_filters(opt)

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
        image = filter_func(np.asarray(dataset))
        if image.ndim == 4:
            assert np.argmin(image.shape) == 0, "Color channel must be in the first dimension"
            if image.shape[0] == 1:
                image = image[0]
            else:
                image = np.moveaxis(image, 0, -1)
                if image.shape[-1] > 3:
                    # reduce to 3 color channels like neuroglancer does
                    channel_selection = image[:, :, :, 2:]
                    channel_selection = np.minimum(np.sum(channel_selection, axis=-1), 255)
                    image[:, :, :, 2] = channel_selection
                    image = image[:, :, :, :3]
                elif image.shape[-1] < 3:
                    # increase to 3 color channels
                    missing = 3 - image.shape[-1]
                    new_values = np.zeros(image.shape[:-1] + (missing,), dtype=np.uint8)
                    image = np.append(image, new_values, axis=-1)
        elif image.ndim == 3:
            image = (label2rgb(image) * 255).astype(np.uint8)
        else:
            raise NotImplementedError(f"Dimensionality {image.ndim} is not implemented")

        image = np.moveaxis(image, 1, 2) # swap x- and y-axis
        dim = image.ndim
        initial_frame = Image.fromarray(image[index(0, dim)])
        additional_frames = [Image.fromarray(image[index(x, dim)]) for x in range(1, image.shape[opt.slice_axis])]
        if opt.add_reverse:
            additional_frames += list(reversed(additional_frames))[1:-1]

        initial_frame.save(path.join(opt.output_dir, get_image_name(dataset.name, i)),
                           format='GIF', append_images=additional_frames, save_all=True,
                           duration=opt.duration, loop=opt.loop)

def save_png(opt):
    f = h5py.File(opt.input_file)
    directories = opt.directories
    os.makedirs(opt.output_dir, exist_ok=True)
    images = []
    filter_func = create_filters(opt)

    for directory in directories:
        images.extend(extract_datasets(f, directory))

    def index(i: int, dim: int):
        return tuple([(slice(None) if x != opt.slice_axis else i) for x in range(dim)])

    def get_image_name(full_dataset_name, image_index):
        if image_index < len(opt.names):
            name = opt.names[image_index]
            if not name.endswith('.png'):
                name += '.png'
            return name
        else:
            dataset_name = full_dataset_name[full_dataset_name.rfind('/') + 1:]
            if dataset_name == '':
                return 'output.png'
            else:
                return dataset_name + '.png'

    for i, dataset in enumerate(images):
        image = filter_func(np.asarray(dataset))
        if image.ndim == 4:
            assert np.argmin(image.shape) == 0, "Color channel must be in the first dimension"
            if image.shape[0] == 1:
                image = image[0]
            else:
                image = np.moveaxis(image, 0, -1)
                if image.shape[-1] > 3:
                    # reduce to 3 color channels like neuroglancer does
                    channel_selection = image[:, :, :, 2:]
                    channel_selection = np.minimum(np.sum(channel_selection, axis=-1), 255)
                    image[:, :, :, 2] = channel_selection
                    image = image[:, :, :, :3]
                elif image.shape[-1] < 3:
                    # increase to 3 color channels
                    missing = 3 - image.shape[-1]
                    new_values = np.zeros(image.shape[:-1] + (missing,), dtype=np.uint8)
                    image = np.append(image, new_values, axis=-1)
        else:
            assert image.ndim == 3, "Images must be either 3D or 4D"

        image = np.moveaxis(image, 1, 2) # swap x- and y-axis
        dim = image.ndim
        image = Image.fromarray(image[index(opt.png_indeces[i], dim)])

        image.save(path.join(opt.output_dir, get_image_name(dataset.name, i)), format='png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Render gifs from an h5 file. For a sample config see the sample_config.py file.')
    parser.add_argument('config', type=str, help='Which config in the ./configs directory to use')
    parser.add_argument('--png', action='store_true', help='If provided, PNGs will be stored at given indeces')
    opt = parser.parse_args()

    config = importlib.import_module(f'configs.{opt.config}').config

    if opt.png:
        save_png(config)
    else:
        main(config)
