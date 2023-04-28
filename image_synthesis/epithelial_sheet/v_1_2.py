import numpy as np
from scipy import ndimage as ndi

import skimage
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

def extract_slice(image, point, shape):
    """
    Returns a slice that can be used to index the given `image` and a slice
    to index a mask of the given `shape`. The `image_slice` has as center the given
    `point`. The size of this slice is the same as `shape` except if it would
    go out of bounds of the image. In this case the slice is cropped and the
    `mask_slice` can be applied to the mask to crop the mask in the same way.
    The third returned value are the coordinates of the point relative to the slice.
    """
    if type(shape) != np.array:
        shape = np.array(shape)
    lower = (point - np.ceil(shape / 2)).astype(int)
    mask_lower = np.where(lower < 0, -lower, 0)
    lower = np.maximum(lower, 0)

    upper = (point + np.floor(shape / 2)).astype(int)
    mask_upper = np.where(upper > image.shape, shape - (upper - image.shape), shape)
    upper = np.minimum(upper, np.array(image.shape))

    image_slice = tuple([slice(lower[i], upper[i]) for i in range(3)])
    mask_slice = tuple([slice(mask_lower[i], mask_upper[i]) for i in range(3)])
    return image_slice, mask_slice, point - lower


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

    blob_image = create_blob_image(cell_image, config)
    logging.info('Created blobs')

    return np.concatenate((np.expand_dims(cell_image, 0), np.expand_dims(blob_image, 0)), axis=0)

def create_blob_image(cell_image, config):
    blob_points = pg.generate_3d_centers(
        limits=np.array([[0, config.limits[2]], [0, config.limits[1]], [0, config.limits[0]]]),
        min_dist=config.blob_config.min_dist,
        max_dist=config.blob_config.max_dist,
        min_child_count=config.blob_config.min_child_count,
        max_child_count=config.blob_config.max_child_count,
        angle_noise=config.blob_config.angle_noise,
        plane_distance=config.blob_config.plane_distance,
        max_offset_from_plane=config.blob_config.max_offset_from_plane,
        first_plane_offset=config.blob_config.first_plane_offset,
        max_center_offset=config.blob_config.max_center_offset,
    )
    blob_points = blob_points.astype(int)
    blob_points[:, [0, 2]] = blob_points[:, [2, 0]]

    logging.info(f'Generated {blob_points.shape[0]} many blob points')

    mean = config.blob_config.mean
    std = config.blob_config.std
    blob_min = config.blob_config.blob_min
    blob_max = config.blob_config.blob_max
    z_scaling = config.blob_config.z_scaling
    max_noise = config.blob_config.max_noise
    noise_resolution = config.blob_config.noise_resolution
    noise_shape = np.array(config.blob_config.noise_shape)

    max_shape = np.array([z_scaling * (blob_max + max_noise * 2), blob_max + max_noise * 2, blob_max + max_noise * 2]) + [2, 2, 2]
    max_shape = max_shape.astype(int)

    assert (noise_shape >= max_shape).all(), \
        f'Provided noise shape {noise_shape} must be at least {max_shape} at all coordinates'

    blob_image = np.zeros(cell_image.shape, dtype=np.uint8)
    for blob in blob_points:
        if cell_image[tuple(blob)] == 0:
            continue

        size = np.clip(np.random.normal(mean, std), blob_min, blob_max)
        shape = np.array([size * z_scaling, size, size]) + [max_noise * z_scaling * 2, max_noise * 2, max_noise * 2] + [2, 2, 2]
        shape = shape.astype(int)
        image_slice, mask_slice, p_relative = extract_slice(cell_image, blob, shape)
        
        image_patch = cell_image[image_slice]
        shape = image_patch.shape

        dist_trans = np.ones(shape, dtype=np.uint8)
        dist_trans[tuple(p_relative)] = 0
        dist_trans = ndi.distance_transform_edt(dist_trans, sampling=(1 / z_scaling, 1, 1))

        noise = perlin.generate_perlin_noise_3d(noise_shape, noise_resolution)
        noise = noise[mask_slice]
        noise = (noise / max(noise.min(), noise.max(), key=abs)) * max_noise
        dist_trans += noise

        blob_mask = dist_trans <= size / 2
        blob_membrane_voxels = dist_trans[blob_mask & (image_patch == 0)].flatten()
        if len(blob_membrane_voxels) != 0:
            blob_mask = dist_trans < blob_membrane_voxels.min()

        if not blob_mask.any():
            continue

        nonzero = np.nonzero(blob_mask)
        sizes = np.array([nonzero[i].max() + 1 - nonzero[i].min() for i in range(3)])

        if sizes.max() >= blob_min:
            blob_image[image_slice][blob_mask] = 255


    return blob_image
