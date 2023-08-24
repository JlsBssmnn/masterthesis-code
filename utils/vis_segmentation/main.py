import argparse
from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.misc import label2rgb

def main(input_file, output_file, dataset):
    with h5py.File(input_file) as f:
        if len(f.keys()) == 1:
            assert dataset is None or f.keys()[0] == dataset, 'The specified dataset does not exist'
            seg = np.asarray(f[list(f.keys())[0]])
        else:
            assert dataset is not None, 'Dataset must be specified when multiple ones are in the h5 file'
            seg = np.asarray(f[dataset])
    
    i = 0
    while seg.ndim > 2:
        index = int(input(f'Image is more than 2D, enter how to index axis {i}: '))
        seg = seg[index]
        i += 1

    rgb = label2rgb(seg)
    fig, ax = plt.subplots()
    ax.imshow(rgb)
    plt.ion()
    fig.show()

    while True:
        seed = input('Type the seed for the next label2rgb, type "exit" to exit: ')
        if seed == 'exit':
            break
        rgb = label2rgb(seg, int(seed))
        ax.imshow(rgb)
        # plt.show(block=False)

    plt.imsave(output_file, rgb)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str, help='The h5 file that stores the segmentation')
    parser.add_argument('output_file', type=str, help='The path to the image file where the segmentation will be stored')
    parser.add_argument('--dataset', default=None, type=str, help='Which dataset in the h5 file to use')
    args = parser.parse_args()

    main(args.input_file, args.output_file, args.dataset)
