import json
import h5py
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage as ndi
import skimage
import numpy as np
import neuroglancer
import webbrowser
from config_tuner.main import library_func
from evaluation.config.template import SegmentationConfig
from evaluation.evaluation_utils import get_path, save_images

from utils.neuroglancer_viewer.neuroglancer_viewer import show_image

#####################################################################################################
################### The following parameters can be tweaked to get a segmentation ###################
#####################################################################################################
    # min_distance: int             - The min_distance parameter for finding the local maxima
    # threshold_abs: float | None   - The threshold_abs parameter for finding the local maxima
    # threshold_rel: float | None   - The threshold_rel parameter for finding the local maxima
    # footprint: npt.NDArray | None - The footprint parameter for finding the local maxima
    # p_norm: float                 - The p_norm parameter for finding the local maxima
    # membrane_threshold: float     - The pixel intensity up to which a pixel definetly is a membrane
#####################################################################################################

def compute_segmentation(image, config, membrane_black):
    # We transform the image to a boolean array: True means cell, False means membrane
    if membrane_black:
        image = image > config['membrane_threshold']
    else:
        image = image < config['membrane_threshold']

    dist = ndi.distance_transform_edt(image)
    peak = peak_local_max(dist, footprint=np.ones(config['peak_footprint_size']), exclude_border=False)

    labels = np.zeros_like(image, dtype=int)

    cost_array = config['cost_func_a'] * np.e**(-config['cost_func_b'] *dist)
    cost_array[~image] = -1

    for i, point in enumerate(peak):
        lower_point = np.array([max(0, point[0] - config['merge_area'][0]), max(0, point[1] - config['merge_area'][1])])
        area = (
            slice(lower_point[0], min(image.shape[0], point[0] + config['merge_area'][0])),
            slice(lower_point[1], min(image.shape[1], point[1] + config['merge_area'][1])),
        )
        point_rel = point - lower_point
        labeled_points = np.transpose(np.nonzero(labels[area] != 0))

        min_cost = float('inf')
        closest_point = None
        for other_point in labeled_points:
            try:
                _, cost = skimage.graph.route_through_array(cost_array, point, other_point + lower_point)
            except:
                continue
            if cost < min_cost:
                closest_point = other_point + lower_point
                min_cost = cost
      
        if min_cost < config['cost_threshold']:
            labels[tuple(point)] = labels[tuple(closest_point)]
        else:
            labels[tuple(point)] = labels.max() + 1

    return watershed(-dist, markers=labels, mask=image)

def create_segmentation(images, config: SegmentationConfig):
    viewer = neuroglancer.Viewer()
    webbrowser.open(str(viewer), new=0, autoraise=True)
    watershed_result = None

    tweak_image = images[config.tweak_image_idx][eval(config.slice_str)]
    show_image(viewer, tweak_image, name='image')
    membrane_black = config.membrane_black

    def tweak_callback(config):
        nonlocal watershed_result
        if membrane_black:
            tresholded_image = tweak_image > config['membrane_threshold']
        else:
            tresholded_image = tweak_image < config['membrane_threshold']

        tresholded_image = np.array([0, 255]).astype(np.uint8)[tresholded_image.astype(int)]
        watershed_result = compute_segmentation(tweak_image, config, membrane_black)

        show_image(viewer, tresholded_image, name='thresholded')
        show_image(viewer, watershed_result.astype(np.uint16), name='labels', segmentation=True)

    final_config = library_func([
        {'attr_name': 'membrane_threshold', 'attr_type': float, 'default': 10},
        {'attr_name': 'peak_footprint_size', 'attr_type': tuple[int], 'default': '15,15'},
        {'attr_name': 'merge_area', 'attr_type': tuple[int], 'default': '20,20'},
        {'attr_name': 'cost_func_a', 'attr_type': float, 'default': 30},
        {'attr_name': 'cost_func_b', 'attr_type': float, 'default': 0.3},
        {'attr_name': 'cost_threshold', 'attr_type': float, 'default': 80},
        ], tweak_callback)

    final_config['tweak_image_idx'] = config.tweak_image_idx
    final_config['membrane_black'] = config.membrane_black

    segmented = []
    for i, image in enumerate(images):
        if i == config.tweak_image_idx:
            segmented.append(watershed_result)
        else:
            segmented_image = compute_segmentation(image[eval(config.slice_str)], final_config, membrane_black)
            segmented.append(segmented_image)
    return segmented, final_config

def one_step(config: SegmentationConfig):
    """
    Can be used to start segmentation from an image file. The result is saved to an output file.
    """
    assert config.input_file is not None and config.output_file is not None
    assert config.config_output_file is not None
    assert config.input_datasets is not None and config.output_datasets is not None and \
            len(config.input_datasets) == len(config.output_datasets)
    assert 0 < len(config.input_datasets) and 0 < len(config.output_datasets)

    images = []
    with h5py.File(config.input_file) as f:
        for dataset in config.input_datasets:
            images.append(np.asarray(f[dataset]))

    segmented, final_config = create_segmentation(images, config)
    save_images(config.output_file, segmented, config.output_datasets)

    config_file = config.config_output_file
    if not config_file.endswith('.json'):
        config_file += '.json'

    with open(get_path(config_file), 'x') as f:
        f.write(json.dumps(final_config, sort_keys=True, indent=4))
