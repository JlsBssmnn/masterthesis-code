from __future__ import annotations
from .Line import Line
from .types import Point
from .LineSegment import LineSegment
import numpy as np
from .config import Viewport, EdgeDecompositionConfig
from scipy.spatial import Voronoi
from .utils import random_float

class OpenSegment(Line):
  def __init__(self, center_point: Point, start_point: Point, normal: Point, direction: Point):
    super().__init__(center_point, start_point)
    self.direction = direction
    self.normal = normal

  @classmethod
  def from_voronoi(cls, v: Voronoi, point_indices: tuple[int, int], center_point: Point, start_point: Point, normal: Point) -> OpenSegment:
    n = np.array([-normal[1], normal[0]])

    midpoint = v.points[list(point_indices)].mean(axis=0)
    center = v.points.mean(axis=0)
    direction = np.sign(np.dot(midpoint - center, n)) * n
    direction /= np.linalg.norm(direction)

    # invert vector if it's facing outwards
    vec_to_center = center_point - start_point
    angle = np.arccos(np.dot(vec_to_center, normal) / (np.linalg.norm(vec_to_center) * np.linalg.norm(normal)))
    if angle > np.pi / 2:
      normal *= [-1, -1]
    normal = normal / np.linalg.norm(normal)

    return cls(center_point, start_point, normal, direction)

  @classmethod
  def from_open_segment(cls, open_segment: OpenSegment, start_point: Point) -> OpenSegment:
    return cls(open_segment.center_point, start_point, open_segment.normal, open_segment.direction)
      
  def has_segment_point(self, point: Point) -> bool:
    return (self.start_point == point).all()

  def change_position(self, change: float):
    self.start_point += self.normal * change

  def get_direction(self):
    return self.direction

  def get_end_point(self, viewport: Viewport) -> Point:
    if not viewport.contains(self.start_point):
      return self.start_point
    border_x = viewport.origin[0] if self.direction[0] < 0 else viewport.origin[0] + viewport.width

    multiplier = (border_x - self.start_point[0]) / self.direction[0]
    end_point = self.start_point + multiplier * self.direction
    if viewport.contains(end_point):
      return end_point

    border_y = viewport.origin[1] if self.direction[1] < 0 else viewport.origin[1] + viewport.height
    multiplier = (border_y - self.start_point[1]) / self.direction[1]
    end_point = self.start_point + multiplier * self.direction
    return end_point

  def create_kink_points(self, config: EdgeDecompositionConfig, viewport: Viewport) -> list[Point]:
    end_point = self.get_end_point(viewport)
    dist = np.linalg.norm(self.start_point - end_point)
    direction = self.direction * dist

    i = 0
    kink_locations: list[float] = []
    while i < dist - config.min_distance:
      kink_locations.append(i / dist)
      i += np.random.random() * (config.max_distance - config.min_distance) + config.min_distance
    
    if len(kink_locations) <= 1:
      return []
    del kink_locations[0]

    return [self.start_point + direction * kink + self.normal * random_float(-config.normal_shift, config.normal_shift) for kink in kink_locations]

  def decompose(self, config: EdgeDecompositionConfig, viewport: Viewport) -> list[Line]:
    new_edges = []
    end_point = self.get_end_point(viewport)
    kink_points = self.create_kink_points(config, viewport)
    if not kink_points:
      return new_edges

    new_edges.append(LineSegment(self.center_point, self.start_point.copy(), kink_points[0]))
    for i in range(len(kink_points) - 1):
      start = kink_points[i]
      end = kink_points[i + 1]
      new_edges.append(LineSegment(self.center_point, start.copy(), end))
    new_edges.append(LineSegment(self.center_point, kink_points[-1].copy(), end_point.copy()))
    new_edges.append(OpenSegment.from_open_segment(self, end_point))

    return new_edges

  def get_end_point_else_start(self):
    return self.start_point

  def __str__(self) -> str:
    return f"OpenSegment(center_point={self.center_point}, start_point={self.start_point}, normal={self.normal}, direction={self.direction}"