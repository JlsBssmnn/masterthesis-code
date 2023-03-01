import numpy as np
from scipy.spatial import Delaunay

def triangulate_shape(points, faces: list[list[int]]):
  """
  Returns an array (n, 3) of the point indices that form
  the triangles of the surface of the shape where n is the
  number of triangles.

  Parameters:
  -------
  points: A 2d array (n, 3) of the 3d coordinates of the n
  points that define the shape.

  faces: A 2d list where each sublist (list in the 2nd dimension)
  holds the indices of the points that belong to one face of the
  shape. (Number of elements in first dimension == number of faces)
  """
  triangles = np.empty((0, 3))
  for face in faces:
    face_points = points[face]
    face_triangles = triangulate_3d_plane(face_points)

    # adjust triangle indices to the indices of all points
    diff = face - np.arange(len(face))
    masks = [face_triangles == i for i in range(len(diff))]
    for i in range(len(diff)):
      face_triangles[masks[i]] += diff[i]

    triangles = np.append(triangles, face_triangles, axis=0)
  
  return triangles

def triangulate_3d_plane(points):
  """
  Triangulates a 3d plane which is defined by the provided points.

  Returns:
  -------
  A 2d array of indices of points that form the triangles.
  """
  assert len(points) >= 3
  assert points.shape[1] == 3
  assert np.unique(points, axis=0).shape[0] == points.shape[0]

  # determine the normal of the plane by taking the direction vectors
  # between 3 points that don't lie in one line
  v1 = points[0] - points[1]
  for i in range(2, len(points)):
    v2 = points[0] - points[i]
    normal = np.cross(v1, v2)

    if (normal != 0).any():
      break
  d = np.dot(normal, points[0])

  # check that all points lie in the same plane
  plane_distances = np.abs(np.dot(points, normal) - d)
  assert (plane_distances < 1e-7).all(), f"Not all points are on the\
    plane, the plane distances are {plane_distances}"

  projection_coord = np.nonzero(normal)[0][0]
  coords = [0, 1, 2]
  coords.remove(projection_coord)
  projected_points = points[:, coords]

  triangulation = Delaunay(projected_points)
  return triangulation.simplices