import argparse
import webbrowser
import sys
import pathlib
import h5py
import numpy as np
import neuroglancer
from skimage import io
import importlib

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.resolve()))
from utils.misc import StoreDictKeyPair
from utils.neuroglancer_viewer.neuroglancer_viewer import show_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', help='Path to the file that contains the image')
    parser.add_argument('--dataset_name', default=None, nargs='+', help='Name of the dataset that contains the image')
    parser.add_argument('--scales', default=[0.5, 0.2, 0.2], nargs=3, type=float, help='Scaling factor for the image')
    parser.add_argument('--slice', default=None, type=str, help='Specify a slice of the image that shall be shown')
    parser.add_argument('--setting', default=None, type=str, help='One of the defined settings to change the settings of neuroglancer')
    parser.add_argument('--setting_options', default=dict(), action=StoreDictKeyPair, nargs="+", metavar="KEY=VAL", help='Options that are passed to the setting')
    parser.add_argument('--segmentation', action='store_true', help='If the image is a segmentation')
    opt = parser.parse_args()

    images = []
    if opt.file_path.endswith('.h5'):
        with h5py.File(opt.file_path) as f:
            datasets = opt.dataset_name if opt.dataset_name is not None else list(f.keys())
            for dataset_name in datasets:
                images.append(np.asarray(f[dataset_name]))
    elif opt.file_path.endswith('.tif') or opt.file_path.endswith('.tiff'):
        datasets = ['image']
        images = [io.imread(opt.file_path)]
    else:
        raise NotImplementedError('File extension not supported')

    if opt.slice is not None:
        for i, image in enumerate(images):
            images[i] = eval(f'images[i][{opt.slice}]') 
    viewer = neuroglancer.Viewer()

    for i, image in enumerate(images):
        show_image(viewer, image, name=datasets[i], segmentation=opt.segmentation, scales=opt.scales)

    if opt.setting is not None:
        setting = importlib.import_module(f'settings.{opt.setting}').setting
        setting(viewer, **opt.setting_options)
    print(viewer)
    webbrowser.open(str(viewer), new=0, autoraise=True)
    input("Done?")
