from typing import Iterable
import numpy as np
import tifffile as tf
import open3d as o3d

def save_voxel_grid(name: str, grid: o3d.geometry.VoxelGrid):
  voxel_positions = np.array([voxel.grid_index for voxel in grid.get_voxels()])
  save_points_array(name, voxel_positions)

def save_voxel_grids(name: str, grids: Iterable[o3d.geometry.VoxelGrid]):
  assert len(grids) > 0
  voxel_positions = None

  for grid in grids:
    points = np.array([voxel.grid_index for voxel in grid.get_voxels()])
    if points.shape[0] == 0:
      continue
    elif voxel_positions is None:
      voxel_positions = points
    else:
      voxel_positions = np.append(voxel_positions, points, axis=0)

  save_points_array(name, voxel_positions)

def save_points_array(name: str, points: Iterable):
  if type(points) != np.ndarray:
    points = np.array(points, dtype=np.uint16)
  elif str(points.dtype).startswith('float'):
    points = np.rint(points).astype(np.uint16)

  # shape = points.max(axis=0) - points.min(axis=0) + 1
  shape = points.max(axis=0) + 1
  img = np.zeros((shape[2], 3, shape[0], shape[1]), dtype='<u2')
  for cord in points:
    img[cord[2], :, cord[0], cord[1]] = 5000
  tf.imwrite(name, img, imagej=True)