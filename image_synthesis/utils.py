import numpy as np
import numpy.typing as npt
from typing import Callable

# The rotation matrix by the angle theta around the x-axis
def rx(theta):
    return np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])

def ry(theta):
    return np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])

def rz(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])

rotation_matricies = np.array([rz, ry, rx])

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
  factor = (plane_d - np.dot(point, plane_normal)) / np.dot(vector, plane_normal)
  return np.linalg.norm(vector * factor)


def bezier_quadratic(p0, p1, p2) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
  p0 = p0.reshape(1, -1)
  p1 = p1.reshape(1, -1)
  p2 = p2.reshape(1, -1)

  def interpolate(t: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    t = t.reshape(-1, 1)
    return p1 + (1-t)**2 * (p0 - p1) + t**2 * (p2 - p1)
  return interpolate