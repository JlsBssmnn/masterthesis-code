import numpy as np
import numpy.typing as npt
from scipy import ndimage as ndi
import perlin_numpy as perlin
import image_synthesis.utils as utils
import cc3d
from image_synthesis.logging_config import logging

class BrainbowGenerator:
    def __init__(self, config):
        logging.info('Creating BrainbowGenerator')
        self.config = config
        self.neuron_count = 0
        np.random.seed(config.seed)

        noise_resolution = (np.array(self.config.image_size) / self.config.noise_resolution_factor).astype(int)
        noise_intensity = ((self.config.max_thickness - self.config.min_thickness) / 2) / self.config.perlin_max
        self.noise = perlin.generate_perlin_noise_3d(self.config.image_size, noise_resolution) * noise_intensity
        self.noise[self.noise > 0] = 0
        logging.info('Generated Perlin Noise')

    def create_images(self):
        image = np.zeros(self.config.image_size, dtype=np.uint16)
        retries = self.config.retries
        self.neuron_count = 0
        logging.info('beginning the neuron generation loop')

        while self.neuron_count < self.config.neuron_count and retries > 0:
            start_point, start_direction = self.create_start_point()
            lines, line_indices = self.create_neuron(start_point, start_direction)
            neuron_image = self.create_neuron_image(lines, line_indices)

            mask = neuron_image == 0
            
            if self.valid_new_neuron(image, mask):
                self.neuron_count += 1
                image[mask] = self.neuron_count
            else:
                retries -= 1

        logging.info('finished generation of synthetic brainbows with a total of %d neurons', self.neuron_count)
        return image

    def valid_new_neuron(self, image, neuron) -> bool:
        """
        Tests whether the given neuron can be inserted into the image without
        breaking an existing neuron into separate pieces.
        """
        new_image = image.copy()
        new_image[neuron] = self.neuron_count + 1

        _, con_components = cc3d.connected_components(new_image, return_N=True, connectivity=6)
        return con_components == self.neuron_count + 1

    def render_lines(self, lines, line_indices):
        image = np.ones(self.config.image_size).astype(np.uint8)
        for i, j in line_indices:
            p = lines[i].copy()
            q = lines[j]
            diff = (q - p).astype(int)
            assert not (diff == 0).all(), \
                    "The start and end point of line must be distinct"

            base = np.abs(diff).argmax()
            m = diff / abs(diff[base])
            for i in range(abs(diff[base]) + 1):
                image[tuple(np.round(p).astype(int))] = 0
                p += m
        return image


    def create_start_point(self):
        start_point = np.random.randint([0, 0, 0], self.config.image_size)
        side_to_project = np.random.randint(0, 3)
        project_value = np.random.choice([0, self.config.image_size[side_to_project] - 1])
        start_point[side_to_project] = project_value

        p = start_point.copy()
        p[side_to_project] += (1 if project_value == 0 else -1)
        for axis in set([0, 1, 2]) - set([side_to_project]):
            p = utils.rotate_point_3d(utils.rotation_matricies[axis]\
                    (utils.random_float(-self.config.max_rotation, self.config.max_rotation)), p, start_point)
        start_direction = utils.norm(p - start_point)
        return start_point, start_direction

    def create_neuron(self, start_point, start_direction):
        lines = [start_point]
        line_indices = []
        queue: list[tuple[int, npt.NDArray]] = [(0, start_direction)]
        while queue:
            p_index, direction = queue[0]
            p = lines[p_index]

            new_point = p + direction * utils.random_float(self.config.min_len, self.config.max_len)
            new_point_index = len(lines)

            point_left_image = False
            if not ((np.array([0, 0, 0]) <= new_point).all() and (np.round(new_point) < np.array(self.config.image_size)).all()):
                point_left_image = True
                distances = []
                for axis in range(3):
                    if direction[axis] == 0:
                      continue
                    normal = np.array([0, 0, 0])
                    normal[axis] = 1
                    plane_d = self.config.image_size[axis] - 1 if direction[axis] > 0 else 0
                    distances.append(utils.point_plane_dist_along_vec(p, direction, normal, plane_d))
                new_point = p + direction * min(distances)

            new_point = np.round(new_point)
            if not (new_point - p == 0).all():
                lines.append(np.round(new_point))
                line_indices.append([p_index, new_point_index])
              
            if point_left_image:
                queue.pop(0)
                continue

            if np.random.rand() < self.config.branch_prob:
                new_direction = direction.copy()
                for axis in range(3):
                    new_direction = utils.rotate_point_3d(utils.rotation_matricies[axis]\
                            (utils.random_float(-self.config.max_rotation, self.config.max_rotation)), new_direction)
                queue.append((p_index, new_direction))
            for axis in range(3):
                direction = utils.rotate_point_3d(utils.rotation_matricies[axis]\
                        (utils.random_float(-self.config.max_rotation, self.config.max_rotation)), direction)
            queue[0] = (new_point_index, direction)

        lines = np.array(lines)
        line_indices = np.array(line_indices)
        return lines, line_indices

    def create_neuron_image(self, lines, line_indicies):
        image = self.render_lines(lines, line_indicies)
        dist_transform = ndi.distance_transform_edt(image)
        dist_transform += self.noise
        image[dist_transform < self.config.min_thickness] = 0
        return image
