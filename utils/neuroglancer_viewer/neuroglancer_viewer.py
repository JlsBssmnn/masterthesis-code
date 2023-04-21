import importlib
import neuroglancer
import numpy as np

def neuroglancer_viewer(images: list, names: list = None, scales: list = None, options = None):
    if names is None:
        names = [f"img_{i}" for i in range(len(images))]

    if scales is None:
        scales = [(1, 1, 1)] * len(images)

    viewer = neuroglancer.Viewer()
    with viewer.txn() as state:
        dimensions = neuroglancer.CoordinateSpace(names=['z', 'y', 'x'],
                                                  units='µm',
                                                  scales=scales[0])
        state.dimensions = dimensions

        for name, img, scale in zip(names, images, scales):
            if img.ndim == 4 and img.shape[0] == 1:
                img = img[0]

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
                            units=['', 'µm', 'µm', 'µm'],
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
                            units='µm',
                            scales=scale)
                    )
                )
            else:
                raise ValueError("Invalid image shape", img.shape)
    
    if options is not None and options.setting is not None:
        import sys
        import pathlib
        sys.path.append(str(pathlib.Path(__file__).parent))
        importlib.import_module(f'settings.{options.setting}').setting(viewer, **options.setting_options)
        del sys.path[-1]

    return viewer
