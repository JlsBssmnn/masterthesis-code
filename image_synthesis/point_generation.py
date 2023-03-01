"""
This file provides functions for generating points for the centers of cells.
"""
from .utils import random_float_array, rx, rotate_points_3d
import numpy as np

def generate_cell_centers_random_points(limits, min_dist, max_dist=None, cell_count=None, max_retries=50):
  """
  Generates an array of points that represent the center of cells. This can be used to
  actually generate synthetic cell images.

  Parameters:
  -------
  limits: An array (d, 2) of limits for the coordinates of the centers, d will be 2 or 3
  depending on whether 2d or 3d images are being produced. The first number in the second
  dimension is the lower bound of that coordinate, the second number is the upper bound.
  min_dist: The minimum distance that 2 cell centers must have
  max_dist: The maximum distance that 2 cell centers must have, note that this might not
  be satisfied if the cell_count is too low
  cell_count: The number of cell centers that shall be generated, note that this might not
  be satisfied if the cell_count and min_dist are too high
  max_retries: The number of times this algorithm will repeat trying to find a point that
  satisfies the provided min_dist after a randomly generated point violated the min_dist
  """
  assert max_dist is not None or cell_count is not None
  if max_dist is None:
    max_dist = 0

  d = limits.shape[0]
  points = np.concatenate((
    np.random.randint(limits[:, 0], limits[:, 1], (1, d)), np.array([[float('inf')]])
  ), axis=1)
  retries = 0

  while points.shape[0] != cell_count and retries < max_retries and points[:, -1].max() > max_dist:
    p = np.random.randint(limits[:, 0], limits[:, 1], (1, d))
    distances = np.linalg.norm(points[:, :-1] - p, axis=1).reshape(-1, 1)
    if distances.min() < min_dist:
      retries += 1
      continue
    points[:, -1] = np.append(points[:, -1:], distances, axis=1).min(axis=1)
    points = np.append(points, np.append(p, distances.min()).reshape(1, -1), axis=0)
    retries = 0

  return points

def generate_cell_centers(limits, min_dist, max_dist, min_child_count, max_child_count, angle_noise,
                         max_center_offset=None):
  """
  Generates an array of points that represent the center of cells. This can be used to
  actually generate synthetic cell images. This algorithm first generates a center in
  the middle and from there rotates around the center to generate further points. This
  procedure is then recursively repeated for the newly generated points.

  Parameters:
  -------
  limits: An array (d, 2) of limits for the coordinates of the centers, d will be 2 or 3
  depending on whether 2d or 3d images are being produced. The first number in the second
  dimension is the lower bound of that coordinate, the second number is the upper bound.
  min_dist: The minimum distance that 2 cell centers must have
  max_dist: The maximum distance that a generated cell has from it's parent
  min_child_count: The minimum number of points that are being generated around a point
  max_child_count: The maximum number of points that are being generated around a point
  angle_noise: The maximum number which is added/subtracted from the angles around a
  point to make the neighboring points not too regular in their positioning (value in
  radians)
  max_center_offset: The initial starting point is the center of the image, moved by a
  random amount. This variable specifies for each dimension the maximum amount the center
  is moved in that dimension.
  """
  if max_dist is None:
    max_dist = 0

  d = limits.shape[0]
  if max_center_offset is None:
    max_center_offset = [0]*d
  assert len(max_center_offset) == d
    
  center = limits.mean(axis=1) + \
    np.random.uniform([0]*d, max_center_offset) * np.random.choice([1, -1], d)
  points = np.concatenate((
    center.reshape(1, -1), np.array([[float('inf')]])
  ), axis=1)
    
  def generate_children(i, depth):
    nonlocal points
    
    child_count = np.random.randint(min_child_count, max_child_count + 1)
    
    if d == 2:
        new_points = np.array([points[i, :-1]]*child_count) + \
            gen_vectors_around_circle(min_dist, max_dist, child_count, angle_noise)
    elif d == 3:
        children_per_circle = int(np.ceil(np.sqrt(child_count)))
        circle_count = int(np.ceil(np.sqrt(child_count)))
        angles = np.linspace(0, np.pi, num=circle_count, endpoint=False)
        new_points = np.empty((0, 3))
        
        for angle in angles:
            points_around_circle = gen_vectors_around_circle(min_dist, max_dist, children_per_circle, angle_noise)
            points_around_circle = np.append(points_around_circle, np.zeros((children_per_circle, 1)), axis=1)
            points_around_circle = rotate_points_3d(rx(angle), points_around_circle)
            new_points = np.append(new_points, points_around_circle, axis=0)
        new_points += np.array([points[i, :-1]]*new_points.shape[0])
    else:
        raise Exception("The dimensionality must be either 2d or 3d")
    
    prior_len = points.shape[0]
    
    # filter points that are outside the limits or too close to another point
    for j in range(new_points.shape[0]):
        p = new_points[j:j+1]
        if not np.logical_and(limits[:, 0] <= p, p <= limits[:, 1]).all():
            continue
        distances = np.linalg.norm(points[:, :-1] - p, axis=1).reshape(-1, 1)
        if distances.min() < min_dist:
            continue
        points[:, -1] = np.append(points[:, -1:], distances, axis=1).min(axis=1)
        points = np.append(points, np.append(p, distances.min()).reshape(1, -1), axis=0)
        
    for j in range(prior_len, points.shape[0]):
        generate_children(j, depth + 1)

  generate_children(0, 0)
    
  return points

def generate_cell_centers_non_recursive(limits, min_dist, max_dist, min_child_count, max_child_count, angle_noise,
                         max_center_offset=None):
  """
  Same as the `generate_cell_centers` function, but it doesn't use recursion 
  """
  if max_dist is None:
    max_dist = 0

  d = limits.shape[0]
  if max_center_offset is None:
    max_center_offset = [0]*d
  assert len(max_center_offset) == d
    
  center = limits.mean(axis=1) + \
    np.random.uniform([0]*d, max_center_offset) * np.random.choice([1, -1], d)
  points = np.concatenate((
    center.reshape(1, -1), np.array([[float('inf')]])
  ), axis=1)

  queue = [0]
    
  while queue:
    i = queue.pop(0)
    
    child_count = np.random.randint(min_child_count, max_child_count + 1)
    
    if d == 2:
        new_points = np.array([points[i, :-1]]*child_count) + \
            gen_vectors_around_circle(min_dist, max_dist, child_count, angle_noise)
    elif d == 3:
        children_per_circle = int(np.ceil(np.sqrt(child_count)))
        circle_count = int(np.ceil(np.sqrt(child_count)))
        angles = np.linspace(0, np.pi, num=circle_count, endpoint=False)
        new_points = np.empty((0, 3))
        
        for angle in angles:
            points_around_circle = gen_vectors_around_circle(min_dist, max_dist, children_per_circle, angle_noise)
            points_around_circle = np.append(points_around_circle, np.zeros((children_per_circle, 1)), axis=1)
            points_around_circle = rotate_points_3d(rx(angle), points_around_circle)
            new_points = np.append(new_points, points_around_circle, axis=0)
        new_points += np.array([points[i, :-1]]*new_points.shape[0])
    else:
        raise Exception("The dimensionality must be either 2d or 3d")
    
    prior_len = points.shape[0]
    
    # filter points that are outside the limits or too close to another point
    for j in range(new_points.shape[0]):
        p = new_points[j:j+1]
        if not np.logical_and(limits[:, 0] <= p, p <= limits[:, 1]).all():
            continue
        distances = np.linalg.norm(points[:, :-1] - p, axis=1).reshape(-1, 1)
        if distances.min() < min_dist:
            continue
        points[:, -1] = np.append(points[:, -1:], distances, axis=1).min(axis=1)
        points = np.append(points, np.append(p, distances.min()).reshape(1, -1), axis=0)
        
    queue[0:0] = list(range(prior_len, points.shape[0]))
    
  return points

def generate_3d_centers(limits, min_dist, max_dist, min_child_count, max_child_count, angle_noise,
                       plane_distance=None, max_offset_from_plane=None, first_plane_offset=None,
                       max_center_offset=None):
    """
    This function generates 3d cell centers by creating 2d planes along one axis, then generating
    points on those planes. The points are then offset by a random amount from the plane.
    """
    assert limits.shape[0] == 3, "This function only works for 3d limits"
    if plane_distance is None or max_offset_from_plane is None:
        plane_distance = min_dist * 3
        max_offset_from_plane = min_dist
        
    if first_plane_offset is None:
        first_plane_offset = max_offset_from_plane
        
    assert max_offset_from_plane < limits[2, 1] - limits[2, 0], "The size along the z-axis must be " + \
        "at least the max_offset_from_plane, s.t. points can be generated"
    
    plane_heights = np.arange(first_plane_offset, limits[2, 1] - limits[2, 0], plane_distance)
    points = np.empty((0, 3))
    for plane_height in plane_heights:
        new_points = generate_cell_centers_non_recursive(
            limits[:-1], min_dist, max_dist, min_child_count, max_child_count, angle_noise, max_center_offset
        )[:, :-1]
        point_count = new_points.shape[0]
        new_points = np.append(new_points, 
                               np.repeat(plane_height, point_count).reshape(-1, 1), axis=1)
        new_points[:, 2] += random_float_array((point_count,),
                                               -max_offset_from_plane, max_offset_from_plane)
        new_points[:, 2] = np.min(np.append(
            np.repeat(limits[2, 1] - 1, point_count).reshape(-1, 1), new_points[:, 2].reshape(-1, 1)
            , axis=1), axis=1)
        new_points[:, 2] = np.max(np.append(
            np.repeat(limits[2, 0], point_count).reshape(-1, 1), new_points[:, 2].reshape(-1, 1)
            , axis=1), axis=1)
        
        points = np.append(points, new_points, axis=0)
    return points

def gen_vectors_around_circle(min_dist, max_dist, child_count, angle_noise):
    start_angle = np.random.rand() * np.pi * 2
    angles = np.linspace(0, np.pi * 2, num=child_count, endpoint=False)
    angles += start_angle
    angles += random_float_array((child_count,), -angle_noise, angle_noise)
    
    lengths = random_float_array((child_count,), min_dist, max_dist)
    mod_angles = angles % (np.pi / 2)
    mods = np.floor(angles / (np.pi / 2)) % 4
    
    adjacent = np.cos(mod_angles) * lengths
    opposite = np.sin(mod_angles) * lengths
    ad_op = np.concatenate((adjacent.reshape(-1, 1), opposite.reshape(-1, 1)), axis=1)
    
    x_change = ad_op[range(ad_op.shape[0]), (mods < 2).astype(int)]
    y_change = ad_op[range(ad_op.shape[0]), (mods >= 2).astype(int)]
    
    x_signs = np.array([1, -1])[np.logical_or(mods == 0, mods == 3).astype(int)]
    y_signs = np.array([1, -1])[np.logical_or(mods == 2, mods == 3).astype(int)]
    
    return (np.concatenate((x_change.reshape(-1, 1), y_change.reshape(-1, 1)), axis=1) * \
         np.concatenate((x_signs.reshape(-1, 1), y_signs.reshape(-1, 1)), axis=1))
