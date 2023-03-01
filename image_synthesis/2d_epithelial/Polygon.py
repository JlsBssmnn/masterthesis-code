from .types import Point
from .Line import Line
from scipy.spatial import Voronoi
import numpy as np
from .LineSegment import LineSegment
from .OpenSegment import OpenSegment
from .utils import in_polygon
from .config import Viewport, EdgeDecompositionConfig

class Polygon:
  edges: list[Line]
  v: Voronoi

  def __init__(self, v: Voronoi, point: int):
    self.edges = []
    pointCount = len(v.ridge_points)
    center_point = v.points[point]
    rp = np.concatenate((v.ridge_points, np.arange(pointCount).reshape((pointCount, 1))), axis=1)
    ridges = rp[np.logical_or.reduce(rp[:, :-1] == point, 1)]
    point_ridges =  ridges[:, :-1]
    ridge_indices = ridges[:, -1].flatten()

    vertex_ridges = [vertex for (i, vertex) in enumerate(v.ridge_vertices) if i in ridge_indices]
    ridges_to_construct = []
    for i, vertex in enumerate(vertex_ridges):
      assert len(vertex) == 2
      assert vertex[0] != -1 or vertex[1] != -1

      if vertex[0] != -1 and vertex[1] != -1:
        self.edges.append(LineSegment(center_point, v.vertices[vertex[0]].copy(), v.vertices[vertex[1]].copy()))
      else:
        ridges_to_construct.append(i)
    
    for i in ridges_to_construct:
      vertices = vertex_ridges[i]
      start_point = v.vertices[vertices[0] if vertices[0] != -1 else vertices[1]]
      normal = v.points[point_ridges[i][0]] - v.points[point_ridges[i][1]]
      self.edges.append(OpenSegment.from_voronoi(v, (point_ridges[i][0], point_ridges[i][1]), center_point, start_point.copy(), normal))

    # sort lines s.t. adjacent lines are adjacent in the list
    for i in range(len(self.edges) - 2, 0, -1):
      intersection = self.edges[i + 1].start_point
      neighbor: Line = next(filter(lambda line: line.has_segment_point(intersection), self.edges))
      neighbor_index = self.edges.index(neighbor)
      self.edges[neighbor_index], self.edges[i] = self.edges[i], self.edges[neighbor_index]

      if (neighbor.start_point == intersection).all():
        neighbor.start_point, neighbor.end_point = neighbor.end_point, neighbor.start_point
    
    if len(self.edges) >= 2 and isinstance(self.edges[0], LineSegment) and (self.edges[1].start_point == self.edges[0].start_point).all():
        self.edges[0].start_point, self.edges[0].end_point = self.edges[0].end_point, self.edges[0].start_point

  def change_size(self, change: float):
    for edge in self.edges:
      edge.change_position(change)
    for i in range(len(self.edges) - 1, -1, -1):
      line1 = self.edges[i]
      line2 = self.edges[i - 1]
      if isinstance(line1, OpenSegment) and isinstance(line2, OpenSegment) and i == 0:
        continue
      p = line1.start_point
      q = line2.start_point if isinstance(self.edges[i - 1], OpenSegment) else line2.end_point

      intersection = line1.get_intersection(line2)
      p[0] = intersection[0]
      p[1] = intersection[1]
      q[0] = intersection[0]
      q[1] = intersection[1]
    self.prune_edges()

  # check if edges can be deleted (e.g. after resizing)
  def prune_edges(self):
    for i in range(len(self.edges) - 2, 0, -1):
      line = self.edges[i]
      line_after = self.edges[i + 1]
      line_before = self.edges[i - 1]

      try:
        intersection = line_after.get_intersection(line_before)
      except ZeroDivisionError:
        continue

      if not in_polygon(intersection, [line.start_point, line.end_point, line.center_point]):
        continue

      line_after.start_point = intersection
      line_before_point = line_before.end_point if isinstance(line_before, LineSegment) else line_before.start_point
      line_before_point[0] = intersection[0]
      line_before_point[1] = intersection[1]

      del self.edges[i]

  def decompose_edges(self, config: EdgeDecompositionConfig, viewport: Viewport):
    for i in range(len(self.edges) - 1, -1, -1):
      new_edges = self.edges[i].decompose(config, viewport)
      if not new_edges:
        continue
      self.edges[i:i+1] = new_edges

