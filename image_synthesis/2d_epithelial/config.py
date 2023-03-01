class Viewport:
  def __init__(self, origin, width: int, height: int):
    assert width > 0 and height > 0
    self.origin = origin
    self.width = width
    self.height = height

  def contains(self, point) -> bool:
    return self.origin[0] <= point[0] <= self.origin[0] + self.width \
      and self.origin[1] <= point[1] <= self.origin[1] + self.height


class EdgeDecompositionConfig:
  def __init__(self, normal_shift: float, min_distance: float, max_distance: float, min_corner_smoothing: float, max_corner_smoothing: float):
    self.normal_shift = normal_shift # how far the new points are maximally away from the original line
    
    # the distance between kink points
    self.min_distance = min_distance
    self.max_distance = max_distance

    # how far to move in start/end of the lines for smoothing the transition between them
    self.min_corner_smoothing = min_corner_smoothing
    self.max_corner_smoothing = max_corner_smoothing