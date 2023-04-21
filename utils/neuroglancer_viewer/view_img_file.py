import argparse
import webbrowser
import sys
import pathlib
import h5py
import numpy as np

sys.path.append(str(pathlib.Path(__file__).parent.parent.resolve()))
from misc import StoreDictKeyPair
from neuroglancer_viewer import neuroglancer_viewer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', help='Path to the file that contains the image')
    parser.add_argument('--dataset_name', default=None, help='Name of the dataset that contains the image')
    parser.add_argument('--scales', default=None, nargs=3, type=float, help='Scaling factor for the image')
    parser.add_argument('--slice', default=None, type=str, help='Specify a slice of the image that shall be shown')
    parser.add_argument('--setting', default=None, type=str, help='One of the defined settings to change the settings of neuroglancer')
    parser.add_argument('--setting_options', default=dict(), action=StoreDictKeyPair, nargs="+", metavar="KEY=VAL", help='Options that are passed to the setting')
    opt = parser.parse_args()

    if opt.file_path.endswith('.h5'):
        if opt.dataset_name is None:
            raise Exception('If h5 is used, --dataset_name must be provided')
        with h5py.File(opt.file_path) as f:
            img = np.asarray(f[opt.dataset_name])
    elif opt.file_path.endswith('.tif') or opt.file_path.endswith('.tiff'):
        img = io.imread(opt.file_path)
    else:
        raise NotImplementedError('File extension not supported')

    if opt.slice is not None:
        img = eval(f'img[{opt.slice}]') 
    viewer = neuroglancer_viewer([img], scales=[tuple(opt.scales)], options=opt)
    print(viewer)
    webbrowser.open(str(viewer), new=0, autoraise=True)
    input("Done?")