import numpy as np
from scipy import ndimage as ndi

from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import skimage.filters as skfilters
import perlin_numpy as perlin

from .. import point_generation as pg
from ..logging_config import logging

def generate_image(config):
    noise_resolution = (np.array(config.limits) / config.noise_resolution_factor).astype(int)
    assert ((config.limits % noise_resolution).astype(int) == 0).all()

    np.random.seed(config.seed)

    points = pg.generate_3d_centers(
        limits=np.array([[0, config.limits[0]], [0, config.limits[1]], [0, config.limits[2]]]),
        min_dist=config.min_dist,
        max_dist=config.max_dist,
        min_child_count=config.min_child_count,
        max_child_count=config.max_child_count,
        angle_noise=config.angle_noise,
        plane_distance=config.plane_distance,
        max_offset_from_plane=config.max_offset_from_plane,
        first_plane_offset=config.first_plane_offset,
        max_center_offset=config.max_center_offset,
    )
    logging.info(f'Generated {points.shape[0]} many starting points')

    black_centers = points.astype(int)
    image = np.ones(config.limits, dtype=bool)
    image[black_centers[:, 0], black_centers[:, 1], black_centers[:, 2]] = 0

    distance = ndi.distance_transform_edt(image)
    logging.info('Computed first distance transform')

    noise = perlin.generate_perlin_noise_3d(config.limits, noise_resolution) * config.noise_intensity
    noisy_distance = distance + noise
    logging.info('Generated perlin noise')

    coords = peak_local_max(-noisy_distance, min_distance=config.min_dist_local_max, exclude_border=False)
    mask = np.zeros(noisy_distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    watershed_result = watershed(noisy_distance, markers, watershed_line=True)
    cell_image = np.array([0, 255]).astype(np.uint8)[(watershed_result != 0).astype(int)]
    logging.info('applied watershed algorithm')

    post_distance = ndi.distance_transform_edt(cell_image)
    post_distance = skfilters.gaussian(post_distance, sigma=config.gaussian_sigma)
    cell_image[post_distance <= config.membrane_threshold] = 0
    logging.info('made membranes thicker')

    return cell_image
