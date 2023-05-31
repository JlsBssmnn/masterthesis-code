import argparse
import h5py
import numpy as np
from scipy.ndimage import label

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This script takes in black-white images of cells segmentation and '
                                     'transforms them into labled images where each cell has a different label')
    parser.add_argument('input_file', type=str, help='The file where the ground truth is stored')
    parser.add_argument('output_file', type=str, help='Where to save the result')
    parser.add_argument('--membrane_label', type=int, default=2, help='The label of the membrane')
    parser.add_argument('--cell_label', type=int, default=3, help='The label of the cell')

    args = parser.parse_args()

    images = {}
    with h5py.File(args.input_file) as f:
        for key in f.keys():
            if type(f[key]) == h5py.Dataset:
                images[key] = np.asarray(f[key])

    f = h5py.File(args.output_file, 'w-')
    for key, image in images.items():
        cell_truth_image = image == args.cell_label
        membrane_truth_image = image != args.membrane_label

        cell_truth_label = label(cell_truth_image)[0]
        membrane_truth_label = label(membrane_truth_image)[0]

        f.create_dataset(f'{key}_cell_truth', data=cell_truth_label)
        f.create_dataset(f'{key}_membrane_truth', data=membrane_truth_label)
    f.close()
