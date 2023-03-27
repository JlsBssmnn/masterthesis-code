import importlib
import neuroglancer
import numpy as np
import argparse
import h5py
from skimage import io

def neuroglancer_viewer(images: list, names: list = None, scales: list = None, options = None):
    if names is None:
        names = [f"img_{i}" for i in range(len(images))]

    if scales is None:
        scales = [(1, 1, 1)] * len(images)

    viewer = neuroglancer.Viewer()
    with viewer.txn() as state:
        dimensions = neuroglancer.CoordinateSpace(names=['z', 'y', 'x'],
                                                  units='nm',
                                                  scales=scales[0])
        state.dimensions = dimensions

        for name, img, scale in zip(names, images, scales):

            if img.ndim == 4:
                if img.shape[-1] == 4:
                    img = np.moveaxis(img, -1, 0)
                else:
                    assert img.shape[0] == 4
                if img.dtype != np.uint8:
                    img = img/np.max(img)
                    img = (img * 255).astype(np.uint8)
                state.layers.append(
                    name=name,
                    layer=neuroglancer.LocalVolume(
                        data=img,
                        dimensions=neuroglancer.CoordinateSpace(
                            names=['c^', 'z', 'y', 'x'],
                            units=['', 'nm', 'nm', 'nm'],
                            scales=(1,) + scale,
                            coordinate_arrays=[
                                neuroglancer.CoordinateArray(labels=['red', 'green', 'blue', 'yellow']), None, None, None,
                            ])
                    ),
                    shader="""
void main() {
  emitRGB(vec3(toNormalized(getDataValue(0)),
               toNormalized(getDataValue(1)),
               toNormalized(getDataValue(2))+toNormalized(getDataValue(3))));
}
""",
                )
            elif img.ndim == 3:
                state.layers.append(
                    name=name,
                    layer=neuroglancer.LocalVolume(
                        data=img,
                        dimensions=neuroglancer.CoordinateSpace(
                            names=['z', 'y', 'x'],
                            units='nm',
                            scales=scale)
                    )
                )
            else:
                raise ValueError("Invalid image shape", img.shape)
    
    if options.setting is not None:
        importlib.import_module(f'settings.{options.setting}').setting(viewer, **options.setting_options)

    return viewer


class StoreDictKeyPair(argparse.Action):
     def __init__(self, option_strings, dest, nargs=None, **kwargs):
         self._nargs = nargs
         super(StoreDictKeyPair, self).__init__(option_strings, dest, nargs=nargs, **kwargs)
     def __call__(self, parser, namespace, values, option_string=None):
         my_dict = {}
         for kv in values:
             k,v = kv.split("=")
             my_dict[k] = v
         setattr(namespace, self.dest, my_dict)


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
    input("Done?")
