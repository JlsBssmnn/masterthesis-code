import numpy as np
from scipy import ndimage as ndi

from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import skimage.filters as skfilters
import perlin_numpy as perlin

from .. import point_generation as pg
from ..logging_config import logging

def get_noise_distribution(config):
    """
    Creates factors for the noise intensity for every z-slice of the image. The values
    are computed from a parabola which peaks at the middle slice. Thus the noise is
    more intensive in the middle and less intensive (value 1 for first and last slice)
    at the sides.

    The parabola has the equation a*x^2 + b*x + c. The parameters a, b and c are computed,
    s.t. the value for the first and last slice is 1 and the peak point of the parabola
    has the `noise_distribution_max` value from the config.
    """
    x1 = 0
    x2 = config.limits[0] - 1
    x3 = (x1 + x2) / 2
    y1 = 1
    y2 = 1
    y3 = config.noise_distribution_max

    a = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / ((x1 - x2) * (x1 - x3) * (x2 - x3))
    b = (x1**2 * (y2 - y3) + x3**2 * (y1 - y2) + x2**2 * (y3 - y1)) / ((x1 - x2) * (x1 - x3) * (x2 - x3))
    c = (x2**2 * (x3 * y1 - x1 * y3) + x2 * (x1**2 * y3 - x3**2 * y1) + x1 * x3 * (x3 - x1) * y2) / ((x1 - x2) * (x1 - x3) * (x2 - x3))

    values = np.arange(config.limits[0])
    return a * values ** 2 + values * b + c


def generate_image(config):
    logging.info('Start image generation')
    noise_resolution = (np.array(config.limits) / config.noise_resolution_factor).astype(int)
    assert ((config.limits % noise_resolution).astype(int) == 0).all()

    np.random.seed(config.seed)

    points = pg.generate_3d_centers(
        limits=np.array([[0, config.limits[2]], [0, config.limits[1]], [0, config.limits[0]]]),
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
    image[black_centers[:, 2], black_centers[:, 1], black_centers[:, 0]] = 0

    distance = ndi.distance_transform_edt(image)
    logging.info('Computed first distance transform')

    noise = perlin.generate_perlin_noise_3d(config.limits, noise_resolution)
    intensity = np.full(config.limits[0], config.noise_intensity, dtype=float)
    intensity *= get_noise_distribution(config)
    noisy_distance = distance + noise * intensity[:, np.newaxis, np.newaxis]
    logging.info('Generated perlin noise')

    coords = peak_local_max(-distance, min_distance=config.min_dist_local_max, exclude_border=False)
    mask = np.zeros(distance.shape, dtype=bool)
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
