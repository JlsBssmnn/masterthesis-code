from .Polygon import Polygon
from .config import Viewport, EdgeDecompositionConfig
from .types import Point
from .Line import Line
from .LineSegment import LineSegment
from .OpenSegment import OpenSegment
from scipy.spatial import Voronoi
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from .utils import rotate_point, vector_angle, random_float, bezier_quadratic

class PolygonCollection:
  polygons: list[Polygon]
  viewport: Viewport

  def __init__(self, v: Voronoi, edge_decomposition_config: EdgeDecompositionConfig, pixel_per_interpolation: int):
    self.polygons = [Polygon(v, i) for i in range(len(v.points))]
    stretch = v.points.ptp(0)
    minimum = v.points.min(0)

    self.viewport = Viewport(minimum - stretch * 0.1, stretch[0] * 1.2, stretch[1] * 1.2)
    self.pixel_per_interpolation = pixel_per_interpolation
    self.edge_decomposition_config = edge_decomposition_config

  def interpolate_line(self, line: Line, start: Point, end: Point) -> list[list[Point]]:
    dist = np.linalg.norm(start - end)

    direction = line.get_direction() * dist
    x_axis = np.array([1, 0])
    angle = vector_angle(direction, x_axis)

    rotated_end = rotate_point(start, angle, np.array([end]))[0]
    temp_line = LineSegment(line.center_point, start, rotated_end)
    kink_points = np.array(temp_line.create_kink_points(self.edge_decomposition_config, self.viewport))

    if len(kink_points) == 0:
      return [[[start[0], start[1]], [end[0], end[1]]]]

    x = np.array([start[0]] + [point[0] for point in kink_points] + [rotated_end[0]])
    y = np.array([start[1]] + [point[1] for point in kink_points] + [rotated_end[1]])
    interpolation = interp1d(x, y, kind='quadratic')

    n_samples = int(np.ceil(dist / self.pixel_per_interpolation))
    # new_x = np.array([start[0] + multiplier * direction[0] for multiplier in np.linspace(0, 1, n_samples, endpoint=False)])
    rotated_x = np.array([start[0] + multiplier * dist for multiplier in np.linspace(0, 1, n_samples, endpoint=False)])
    points = np.concatenate((rotated_x.reshape((-1, 1)), interpolation(rotated_x).reshape(-1, 1)), axis=1)
    points = rotate_point(start, -angle, points)
    lines = []

    for i, point in enumerate(points):
      if i == len(points) - 1:
        lines.append([[point[0], point[1]], [end[0], end[1]]])
      else:
        lines.append([[point[0], point[1]], [points[i+1, 0], points[i+1, 1]]])
    return lines

  def interpolate_edge(self, start: Point, end: Point, intersection: Point) -> list[Point]:
    dist = np.linalg.norm(start - intersection) + np.linalg.norm(intersection - end)
    n_samples = max(int(np.ceil(dist / self.pixel_per_interpolation)), 2)
    t = np.linspace(0, 1, n_samples)
    interpolated = bezier_quadratic(start, intersection, end)(t)
    return [[[x[0], x[1]], [interpolated[i + 1, 0], interpolated[i + 1, 1]]] for i, x in enumerate(interpolated) if i != len(interpolated) - 1]

  def plot(self):
    lines = []
    fig, ax = plt.subplots()
    for j, p in enumerate(self.polygons):
      first_edge_start = None
      neighbor_start = None
      printed = 0

      for i, line in enumerate(p.edges):
        line_length = np.linalg.norm(line.start_point - line.get_end_point(self.viewport))
        smoothing_length = min(line_length / 2, random_float(self.edge_decomposition_config.min_corner_smoothing, self.edge_decomposition_config.max_corner_smoothing))
        if isinstance(line, OpenSegment):
          current_end = line.start_point + line.get_direction() * smoothing_length
        else:
          current_end = line.end_point - line.get_direction() * smoothing_length

        if neighbor_start is None:
          smoothing_length = min(line_length / 2, random_float(self.edge_decomposition_config.min_corner_smoothing, self.edge_decomposition_config.max_corner_smoothing))
          current_start = line.start_point + line.get_direction() * smoothing_length
          first_edge_start = current_start
        else:
          current_start = neighbor_start

        if i == len(p.edges) - 1:
          neighbor_start = first_edge_start
          next_line = p.edges[0]
        else:
          next_line = p.edges[i+1]
          next_line_length = np.linalg.norm(next_line.start_point - next_line.get_end_point(self.viewport))
          next_smoothing_length = min(next_line_length / 2, random_float(self.edge_decomposition_config.min_corner_smoothing, self.edge_decomposition_config.max_corner_smoothing))
          neighbor_start = next_line.start_point + next_line.get_direction() * next_smoothing_length

        if isinstance(line, OpenSegment) or np.linalg.norm(current_start - current_end) > 1e-10:
          lines.extend(self.interpolate_line(line, current_start, current_end if isinstance(line, LineSegment) else line.get_end_point(self.viewport)))

        if isinstance(line, LineSegment):
          lines.extend(self.interpolate_edge(current_end, neighbor_start, next_line.start_point))
        elif i != len(p.edges) - 1:
          lines.extend(self.interpolate_edge(current_start, neighbor_start, next_line.start_point))
        printed -= 1

    ax.add_collection(LineCollection(lines))
    ax.set_xlim(self.viewport.origin[0], self.viewport.origin[0] + self.viewport.width)
    ax.set_ylim(self.viewport.origin[1], self.viewport.origin[1] + self.viewport.height)
    plt.show()

  def decompose_edges(self, config: EdgeDecompositionConfig, viewport: Viewport):
    for polygon in self.polygons:
      polygon.decompose_edges(config, viewport)