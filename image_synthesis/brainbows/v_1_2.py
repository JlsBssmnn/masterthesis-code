from typing import cast
import cc3d
import numpy as np
import numpy.typing as npt
from scipy import ndimage as ndi
import perlin_numpy as perlin
import skimage
import image_synthesis.utils as utils
from image_synthesis.logging_config import logging

class DirectionRotater:
    def __init__(self, config):
        self.config = config

    def rotate_direction(self, direction, rotation):
        for axis in range(3):
            angle = utils.random_float(-rotation, rotation)
            direction = utils.rotate_point_3d(utils.rotation_matrices[axis](angle), direction)
        return direction

    def rotate_direction_min_max(self, direction, min_rotation, max_rotation):
        for axis in range(3):
            angle = utils.random_float(min_rotation, max_rotation)
            angle *= -1 if np.random.rand() < 0.5 else 1
            direction = utils.rotate_point_3d(utils.rotation_matrices[axis](angle), direction)
        return direction

    def rotate_direction_large_small(self, direction):
        if np.random.rand() < self.config.large_rotation_prob:
            rotation = self.config.max_rotation_large
        else:
            rotation = self.config.max_rotation_small
        return self.rotate_direction(direction, rotation)

    def rotate_direction_large(self, direction):
        return self.rotate_direction(direction, self.config.max_rotation_large)

    def rotate_direction_small(self, direction):
        return self.rotate_direction(direction, self.config.max_rotation_small)

    def rotate_branch(self, direction):
        return self.rotate_direction_min_max(direction, self.config.min_branch_rotation, self.config.max_branch_rotation)

class BrainbowGenerator:
    def __init__(self, config):
        logging.info('Creating BrainbowGenerator')
        assert config.max_branches > 1
        self.config = config
        self.neuron_count = 0
        np.random.seed(config.seed)

        noise_resolution = (np.array(self.config.image_size) / self.config.noise_resolution_factor).astype(int)
        self.noise = perlin.generate_perlin_noise_3d(self.config.image_size, noise_resolution)
        self.noise /= max(abs(self.noise.max()), abs(self.noise.min()))

        anti_noise = perlin.generate_perlin_noise_3d(self.config.image_size, noise_resolution)
        anti_noise /= max(abs(anti_noise.max()), abs(anti_noise.min()))

        refinement_noise_resolution = (np.array(self.config.image_size) / self.config.refinement_noise_resolution_factor).astype(int)
        self.refinement_noise = perlin.generate_perlin_noise_3d(self.config.image_size, refinement_noise_resolution)
        self.refinement_noise /= max(abs(self.refinement_noise.max()), abs(self.refinement_noise.min()))
        self.refinement_noise[self.refinement_noise < 0] = 0

        anti_noise[anti_noise > 0] = 0
        self.noise -= anti_noise
        self.noise[self.noise > 0] = 0

        self.direction_rotater = DirectionRotater(config)
        logging.info('Generated Perlin Noise')

    def create_images(self):
        self.image = np.zeros(self.config.image_size, dtype=np.uint16)
        self.neuron_buffer = np.empty(self.config.image_size, dtype=np.uint8)
        retries = self.config.retries
        self.neuron_count = 0
        logging.info('beginning the neuron generation loop')

        while self.neuron_count < self.config.neuron_count and retries > 0:
            start_point, start_direction = self.create_start_point()
            self.create_neuron(start_point, start_direction)
            self.create_neuron_image()

            if not self.insert_new_neuron():
                retries -= 1

        logging.info('finished generation of synthetic brainbows with a total of %d neurons', self.neuron_count)
        return self.image

    def insert_new_neuron(self) -> bool:
        """
        Inserts the neuron in the `neuron_buffer` into the `image`. If this fails,
        the function returns `false`, otherwise `true`.
        """
        mask = self.neuron_buffer == 0
        
        if mask.sum() < self.config.min_voxel_per_neuron:
            return False

        con_components, component_count = cc3d.connected_components(mask, return_N=True, connectivity=6)
        if component_count == 1:
            self.neuron_count += 1
            self.image[mask] = self.neuron_count
            return True

        # if the neuron is disconnected take the largest part of the neuron
        largest_component = None
        largest_component_size = -1
        for i in range(1, component_count + 1):
            component_mask = con_components == i
            size = component_mask.sum()
            if size > largest_component_size:
                largest_component_size = size
                largest_component = component_mask
        if largest_component_size < self.config.min_voxel_per_neuron:
            return False
        self.neuron_count += 1
        self.image[largest_component] = self.neuron_count
        return True

    def draw_line(self, start, end) -> npt.NDArray | None:
        """
        Tries to insert a line from `start` to `end` into the `neuron_buffer`.
        If the neuron would collide with another, this function either terminates
        the neuron (and returns `None`) or moves it around the other neuron and
        returns the new end point that was determined in the process.
        """
        line = skimage.draw.line_nd(start, end, endpoint=True)
        if (self.image[line] == 0).all():
            self.neuron_buffer[line] = 0
            return end

        direction = end - start
        distance_to_border = utils.point_box_dist_along_vec(start, direction, (0, 0, 0), self.config.image_size)
        direction *= distance_to_border / np.linalg.norm(direction)
        p_border = np.round(start + direction).astype(int)

        line = skimage.draw.line_nd(start, p_border, endpoint=True)
        line_transposed = np.transpose(line)

        # edge case: the new line might not touch another neuron (because of floating point arithmetic)
        if (self.image[line] == 0).all():
            # draw a small portion of the line
            length = np.random.randint(self.config.min_len, self.config.max_len + 1)
            self.neuron_buffer[line[:length]] = 0
            return line_transposed[length - 1]

        # the index for the line array where the first voxel of another neuron is located
        first_neuron_pos = np.nonzero(self.image[line] != 0)[0][0] 

        # draw just until the other neuron
        until_neuron = skimage.draw.line_nd(start, line_transposed[first_neuron_pos], endpoint=False)
        self.neuron_buffer[until_neuron] = 0

        # (again index) where the first non-neuron voxel behind the found neuron is located
        first_free_pos = np.where(self.image[line][first_neuron_pos:] == 0)[0]
        if len(first_free_pos) == 0 or np.random.rand() < self.config.termination_probability:
            # there is no free voxel after the neuron (neuron is rigth at the border)
            return None
        first_free_pos = first_free_pos[0] + first_neuron_pos

        start = line_transposed[first_neuron_pos - 1]
        end = line_transposed[first_free_pos]
        search_area, search_origin = self.shortest_path_search_area(start, end)

        cost_array = self.image[search_area].astype(np.int32)
        cost_array[cost_array > 0] = -1 # search will ignore paths with negative numbers
        cost_array[cost_array == 0] = 1 # going 1 voxel through space is cost 1

        try:
            path = skimage.graph.route_through_array(cost_array, start - search_origin, end - search_origin)
        except ValueError as e:
            # there is no path from start to end
            assert str(e) == 'no minimum-cost path was found to the specified end point'
            return None

        path = np.array(path[0]) + search_origin
        self.neuron_buffer[(path[:, 0], path[:, 1], path[:, 2])] = 0
        return end

    
    def shortest_path_search_area(self, start, end):
        """
        Determines an area around the start and end points that can be searched for a
        shortest path. The area is usually much smaller than the entire image, resulting
        in better performance and prevents unnecessary memory consumption.

        Returns
        -------
        index: An index into the `image` array, that can be used to extract the search area
        origin: The origin point of the search area within the `image`. This is the point with
            the smallest coordinates across all dimensions within the search area
        """
        dist = np.abs(start - end)
        missing_size = self.config.shortest_path_search_area - dist
        missing_size = np.round(missing_size / 2).astype(int)
        missing_size[missing_size < 0] = 0

        mins = np.minimum(start, end)
        maxes = np.maximum(start, end) + 1

        mins = np.maximum(mins - missing_size, [0, 0, 0])
        maxes = np.minimum(maxes + missing_size, self.config.image_size)
        return tuple([slice(mins[i], maxes[i]) for i in range(3)]), mins


    def create_start_point(self):
        start_point = np.random.randint([0, 0, 0], self.config.image_size)
        side_to_project = np.random.randint(0, 3)
        project_value = np.random.choice([0, self.config.image_size[side_to_project] - 1])
        start_point[side_to_project] = project_value

        p = start_point.copy()
        p[side_to_project] += (1 if project_value == 0 else -1)
        for axis in set([0, 1, 2]) - set([side_to_project]):
            p = utils.rotate_point_3d(utils.rotation_matrices[axis]\
                    (utils.random_float(-self.config.max_rotation_large, self.config.max_rotation_large)), p, start_point)
        start_direction = utils.norm(p - start_point)
        return start_point, start_direction

    def create_neuron(self, start_point, start_direction):
        """
        Creates a neuron by writing into the `neuron_buffer` variable. All zero values represent the
        neuron while all one values represent empty space. This function will first wipe the
        `neuron_buffer` by setting all values to 1.
        """
        self.neuron_buffer[:,:,:] = 1
        queue: list[tuple[npt.NDArray, npt.NDArray]] = [(start_point, start_direction)]
        total_branch_count = 1
        allowed_to_branch = np.random.rand() < self.config.branching_neuron_prob
        branch_count = np.random.choice(range(2, self.config.max_branches + 1))

        while queue:
            p, direction = queue[0]
            new_point = p + direction * utils.random_float(self.config.min_len, self.config.max_len)

            point_left_image = False
            if not ((np.array([0, 0, 0]) <= new_point).all() and (np.round(new_point) < np.array(self.config.image_size)).all()):
                point_left_image = True
                distance_to_border = utils.point_box_dist_along_vec(p, direction, (0, 0, 0), self.config.image_size)
                new_point = p + direction * distance_to_border

            new_point = np.round(new_point)
            if not (new_point - p == 0).all():
                new_point = self.draw_line(p, new_point)
              
            if point_left_image or new_point is None:
                queue.pop(0)
                continue

            if allowed_to_branch and total_branch_count < branch_count and \
                    np.random.rand() < self.config.branch_prob / total_branch_count:
                total_branch_count += 1
                new_direction = direction.copy()
                new_direction = self.direction_rotater.rotate_branch(new_direction)
                queue.append((p, new_direction))

            direction = self.direction_rotater.rotate_direction_large_small(direction)
            queue[0] = (new_point, direction)

    def create_neuron_image(self):
        if type(self.config.min_thickness) == list:
            min_thickness = np.random.choice(self.config.min_thickness, p = self.config.min_thickness_probabilities)
            largest_min_thickness = max(self.config.min_thickness)
        else:
            min_thickness = self.config.min_thickness
            largest_min_thickness = self.config.min_thickness

        dist_transform = cast(npt.NDArray, ndi.distance_transform_edt(self.neuron_buffer, sampling=[1, 1, 1]))
        free_space = self.image == 0
        self.neuron_buffer[(dist_transform <= 1) & free_space] = 0

        if type(self.config.max_thickness) == list:
            max_thickness = np.random.choice(self.config.max_thickness, p=self.config.max_thickness_probabilities)
        else:
            max_thickness = self.config.max_thickness
        noise_intensity = (max_thickness - largest_min_thickness) / 2

        noise = self.noise * noise_intensity - self.refinement_noise * self.config.refinement_noise_intensity
        dist_transform += noise 
        self.neuron_buffer[(dist_transform < min_thickness) & free_space] = 0
