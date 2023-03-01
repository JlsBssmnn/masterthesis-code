from __future__ import annotations
from .types import Point
from abc import ABC, abstractmethod
from .config import Viewport, EdgeDecompositionConfig

class Line(ABC):
  def __init__(self, center_point: Point, start_point: Point):
    self.center_point = center_point
    self.start_point = start_point
  
  def get_intersection(self, other: Line) -> Point:
    p = self.start_point
    px, py = p
    vx, vy = self.get_direction()

    q = other.get_end_point_else_start()
    qx, qy = q
    w = other.get_direction()
    wx, wy = w

    multiplier = (vy * (qx - px) + py * vx - qy * vx) / (vx * wy - vy * wx)
    return q + multiplier * w

  @abstractmethod
  def get_end_point(self, viewport: Viewport) -> Point:
    pass

  @abstractmethod
  def get_direction(self) -> Point:
    pass

  @abstractmethod
  def change_position(self, change: float):
    pass

  @abstractmethod
  def has_segment_point(self, point: Point) -> bool:
    pass

  @abstractmethod
  def create_kink_points(self, config: EdgeDecompositionConfig, viewport: Viewport) -> list[Point]:
    pass
  
  @abstractmethod
  def decompose(self, config: EdgeDecompositionConfig, viewport: Viewport) -> list[Line]:
    pass

  @abstractmethod
  def get_end_point_else_start(self) -> Point:
    '''Returns the end_point for LineSegments and the start_point for OpenSegments'''
    pass

print("hello again")