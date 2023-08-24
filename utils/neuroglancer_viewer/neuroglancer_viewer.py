import importlib
import neuroglancer
import numpy as np
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))
from utils.misc import scale_image

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
                channel_index = np.array(img.shape).argmin()
                if channel_index == img.ndim - 1:
                    # The channel is at the end
                    img = np.moveaxis(img, -1, 0)
                else:
                    assert channel_index == 0
                if img.dtype != np.uint8:
                    img = img/np.max(img)
                    img = (img * 255).astype(np.uint8)
                if img.shape[channel_index] == 4:
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
                elif img.shape[channel_index] == 3:
                    state.layers.append(
                        name=name,
                        layer=neuroglancer.LocalVolume(
                            data=img,
                            dimensions=neuroglancer.CoordinateSpace(
                                names=['c^', 'z', 'y', 'x'],
                                units=['', 'µm', 'µm', 'µm'],
                                scales=(1,) + scale,
                                coordinate_arrays=[
                                    neuroglancer.CoordinateArray(labels=['red', 'green', 'blue']), None, None, None,
                                ])
                        ),
                        shader="""
    void main() {
    emitRGB(vec3(toNormalized(getDataValue(0)),
                toNormalized(getDataValue(1)),
                toNormalized(getDataValue(2))));
    }
    """,
                    )
                elif img.shape[channel_index] == 2:
                    state.layers.append(
                        name=name,
                        layer=neuroglancer.LocalVolume(
                            data=img,
                            dimensions=neuroglancer.CoordinateSpace(
                                names=['c^', 'z', 'y', 'x'],
                                units=['', 'µm', 'µm', 'µm'],
                                scales=(1,) + scale,
                                coordinate_arrays=[
                                    neuroglancer.CoordinateArray(labels=['red', 'green']), None, None, None,
                                ])
                        ),
                        shader="""
    void main() {
    emitRGB(vec3(toNormalized(getDataValue(0)),
                toNormalized(getDataValue(1)), 0));
    }
    """,
                    )
                else:
                    raise Exception(f"Image with {img.shape[channel_index]} channels not supported")
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

def get_shader(n_channels):
    rgb_values = [f"toNormalized(getDataValue({i}))" for i in range(n_channels)]
    if n_channels > 3:
        rgb_values[2] = '+'.join(rgb_values[2:])
        rgb_values = rgb_values[:3]
    elif n_channels < 3:
        rgb_values.extend(['0'] * (3-n_channels))
    s = ', '.join(rgb_values)

    return """
    void main() {
    emitRGB(vec3(%s));
    }
    """ % s

def show_image(viewer, image, name='image', segmentation=False, scales=[0.5, 0.2, 0.2]):
    with viewer.txn() as state:
        dimensions = neuroglancer.CoordinateSpace(names=['z', 'y', 'x'],
                                                units='nm',
                                                scales=scales)
        state.dimensions = dimensions

        if image.dtype == np.uint16 and not segmentation:
            image = scale_image(image, 0, image.max(), 0, 255).astype(np.uint8)

        argmin = np.argmin(image.shape) != 0
        if image.ndim == 4 and image.shape[0] == 1:
            image = image[0]
        elif image.ndim == 4 and argmin:
            image = np.rollaxis(image, -1)

        if name in state.layers:
            del state.layers[name]
        if image.ndim == 3:
            if image.shape[0] == 3:
                state.layers.append(
                    name=name,
                    layer=neuroglancer.LocalVolume(
                        data=image,
                        dimensions=neuroglancer.CoordinateSpace(
                            names=['c^', 'x', 'y'],
                            units=['', 'nm', 'nm'],
                            scales=[1] + scales[-2:],
                            coordinate_arrays=[
                                neuroglancer.CoordinateArray(labels=['red', 'green', 'blue']), None, None, None,
                            ])),
                            shader="""
    void main() {
    emitRGB(vec3(toNormalized(getDataValue(0)),
                toNormalized(getDataValue(1)),
                toNormalized(getDataValue(2))));
    }
    """,
                )
            else:
                state.layers.append(
                    name=name,
                    layer=neuroglancer.LocalVolume(
                        data=image,
                        dimensions=neuroglancer.CoordinateSpace(
                            names=['z', 'y', 'x'],
                            units='nm',
                            scales=scales)
                    )
                )
        elif image.ndim == 4:
            c = image.shape[0]
            state.layers.append(
                name=name,
                layer=neuroglancer.LocalVolume(
                    data=image,
                    dimensions=neuroglancer.CoordinateSpace(
                        names=['c^', 'z', 'y', 'x'],
                        units=['', 'nm', 'nm', 'nm'],
                        scales=[1] + scales,
                        coordinate_arrays=[
                            neuroglancer.CoordinateArray(labels=['red', 'green', 'blue', 'yellow'][:c]), None, None, None,
                        ])
                ),
                shader=get_shader(c)
            )
        elif image.ndim == 2:
            state.layers.append(
                name=name,
                layer=neuroglancer.LocalVolume(
                    data=image,
                    dimensions=neuroglancer.CoordinateSpace(
                        names=['x', 'y'],
                        units='nm',
                        scales=scales[-2:])
                )
            )
        state.show_slices = False
        state.cross_section_background_color = "#2e2e2e"
        state.layout = "yz"
        if segmentation:
            state.layers[name].type = 'segmentation'
