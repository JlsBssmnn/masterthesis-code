import numpy as np
import numpy.typing as npt
from typing import Callable
import torch
from torchvision import transforms

# The rotation matrix by the angle theta around the x-axis
def rx(theta):
    return np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])

def ry(theta):
    return np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])

def rz(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])

rotation_matrices = np.array([rz, ry, rx])

def norm(vector):
  return vector / np.linalg.norm(vector)

def in_polygon(point, edge_vertices: list) -> bool:
  n = len(edge_vertices)
  sign = None

  for i in range(n):
    p1 = edge_vertices[i]
    p2 = edge_vertices[(i + 1) % n]
    
    v1 = p2 - p1
    v2 = point - p1
    s = np.sign(np.cross(v1, v2))
    if sign is None:
      sign = s
    elif s == 0:
      return False
    elif sign != s:
      return False
  return True


def random_float(lower: float = 0, upper: float = 1) -> float:
  return np.random.random() * (upper - lower)  + lower

def random_float_array(shape, lower: float = 0, upper: float = 1):
    return np.random.rand(*shape) * (upper - lower) + lower

def random_int_array(shape, lower: int = 0, upper: int = 1):
    return np.round(np.random.rand(*shape) * (upper - lower) + lower).astype(int)

def vector_angle(v1, v2) -> float:
  cross = np.cross(v1, v2)
  dot = np.dot(v1, v2)
  return np.arctan2(np.linalg.norm(cross), dot)


def rotate_point(origin, angle: float, points: npt.NDArray[np.float64]):
  return np.array([
    np.cos(angle) * (points[:, 0] - origin[0]) - np.sin(angle) * (points[:, 1] - origin[1]) + origin[0],
    np.sin(angle) * (points[:, 0] - origin[0]) + np.cos(angle) * (points[:, 1] - origin[1]) + origin[1]
  ]).T

def rotate_point_3d(rotation_matrix, point, origin=np.array([0, 0, 0])):
  tmp = point - origin
  tmp = np.dot(rotation_matrix, tmp)
  return tmp + origin

def rotate_points_3d(rotation_matrix, points):
     return np.array([np.dot(rotation_matrix, p) for p in points])


def point_plane_dist_along_vec(point, vector, plane_normal, plane_d):
  """
  Returns the distance from the given point along the given vector to the
  given plane.
  """
  factor = (plane_d - np.dot(point, plane_normal)) / np.dot(vector, plane_normal)
  return np.linalg.norm(vector * factor)


def point_box_dist_along_vec(point, vector, origin, side_lengths):
  """
  Returns the distance from the given point along the given vector to the
  border of the given box. The box is defined by it's origin point (3D) and
  a 3-element iterable that contains the lengths of the sides for each
  dimension. The length is the number of voxels along that dimension, meaning
  adding the length to the origin would be one voxel beyond of the image array.
  """
  distances = []
  for axis in range(3):
    if vector[axis] == 0:
      continue
    normal = np.array([0, 0, 0])
    normal[axis] = 1

    if vector[axis] > 0:
      plane_d = origin[axis] + side_lengths[axis] - 1
    else:
      plane_d = origin[axis]
    distances.append(point_plane_dist_along_vec(point, vector, normal, plane_d))
  assert len(distances) > 0, \
          'It was not possible to determine the distance from the point to the box'
  return min(distances)


def slice_by_plane(shape, normal, d):
    """
    Returns the indices into an array of the given shape that are going along
    the given plane.
    """
    dim = int(np.argmax(np.abs(normal)))
    base_dim = set(range(3))
    base_dim.remove(dim)
    base_dim = list(base_dim)

    axis1, axis2 = np.meshgrid(np.arange(shape[base_dim[0]]), np.arange(shape[base_dim[1]]))
    axis3 = np.rint(np.minimum( \
            (d - normal[base_dim[0]] * axis1 - normal[base_dim[1]] * axis2) / normal[dim], \
            shape[dim] - 1)) \
        .astype(int)

    r_value = [axis1, axis2]
    r_value.insert(dim, axis3)

    return tuple(r_value)

def bezier_quadratic(p0, p1, p2) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
  p0 = p0.reshape(1, -1)
  p1 = p1.reshape(1, -1)
  p2 = p2.reshape(1, -1)

  def interpolate(t: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    t = t.reshape(-1, 1)
    return p1 + (1-t)**2 * (p0 - p1) + t**2 * (p2 - p1)
  return interpolate


def scale_image(image, scale):
  """
  Scales an image given as an array. Scale needs to be an iterable giving the
  scaling along each dimension, e.g. scale = [1, 2, 3] will keep the image the
  same size along the first dimension, double it's size along the second and
  tripple the size along the last.
  """
  dtype = str(image.dtype)
  if type(image) == np.ndarray and dtype.startswith('u'):
      image = image.astype(f'int{2 * int(dtype[4:])}')
  torch_image = torch.tensor(image)
  if scale[0] != 1:
    torch_image = torch.movedim(torch_image, -3, -1)
    size = tuple(torch.tensor(torch_image.shape[-2:]) * torch.tensor([1, scale[0]]))
    torch_image = transforms.Resize(size, transforms.InterpolationMode.NEAREST)(torch_image)
    torch_image = torch.movedim(torch_image, -1, -3)
  size = tuple(torch.tensor(torch_image.shape[-2:]) * torch.tensor(scale[1:]))
  torch_image = transforms.Resize(size, transforms.InterpolationMode.NEAREST)(torch_image)
  return torch_image.detach().cpu().numpy().astype(np.uint16)
