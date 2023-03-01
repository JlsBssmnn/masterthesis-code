from __future__ import annotations
from .Line import Line
import numpy as np
from .config import Viewport, EdgeDecompositionConfig
from ..utils import random_float

class LineSegment(Line):
  def __init__(self, center_point, start_point, end_point):
    super().__init__(center_point, start_point)
    self.end_point = end_point

    diff = start_point - end_point
    normal = np.array([-diff[1], diff[0]])

    # invert vector if it's facing outwards
    vec_to_center = center_point - start_point
    angle = np.arccos(np.dot(vec_to_center, normal) / (np.linalg.norm(vec_to_center) * np.linalg.norm(normal)))
    if angle > np.pi / 2:
      normal *= [-1, -1]
    self.normal = normal / np.linalg.norm(normal)
      
  # return whether the provided point is one of the segment points
  def has_segment_point(self, point) -> bool:
    return (self.start_point == point).all() or (self.end_point == point).all()

  def get_end_point(self, viewport: Viewport):
    return self.end_point

  def change_position(self, change: float):
    self.start_point += self.normal * change
    self.end_point += self.normal * change

  def get_direction(self):
    diff = self.end_point - self.start_point
    if (diff == [0, 0]).all():
      return diff
    else:
      return diff / np.linalg.norm(diff)

  def create_kink_points(self, config: EdgeDecompositionConfig, viewport: Viewport) -> list:
    direction = self.end_point - self.start_point
    dist = np.linalg.norm(direction)

    i = 0
    kink_locations: list[float] = []
    while i < dist - config.min_distance:
      kink_locations.append(i / dist)
      i += np.random.random() * (config.max_distance - config.min_distance) + config.min_distance
    
    if len(kink_locations) <= 1:
      return []
    del kink_locations[0]

    return [self.start_point + direction * kink + self.normal * random_float(-config.normal_shift, config.normal_shift) for kink in kink_locations]

  def decompose(self, config: EdgeDecompositionConfig, viewport: Viewport) -> list[LineSegment]:
    new_edges = []
    kink_points = self.create_kink_points(config, viewport)
    if not kink_points:
      return new_edges

    new_edges.append(LineSegment(self.center_point, self.start_point.copy(), kink_points[0]))
    for i in range(len(kink_points) - 1):
      start = kink_points[i]
      end = kink_points[i + 1]
      new_edges.append(LineSegment(self.center_point, start.copy(), end))
    new_edges.append(LineSegment(self.center_point, kink_points[-1].copy(), self.end_point.copy()))

    return new_edges

  def get_end_point_else_start(self):
    return self.end_point

  def __str__(self) -> str:
    return f"LineSegment(center_point={self.center_point}, start_point={self.start_point}, end_point={self.end_point}"