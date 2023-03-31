import unittest
import numpy as np

from image_synthesis.brainbows.v_1_3 import BrainbowGenerator
from image_synthesis.logging_config import logging

logging.getLogger().setLevel(logging.ERROR)

class TestConfig:
    image_size = (10, 10, 10)
    max_branches = 2
    max_thickness = 20
    min_thickness = 2
    min_voxel_per_neuron = 0
    noise_resolution_factor = 1
    perlin_max = 0.7
    refinement_noise_resolution_factor = 2
    seed = 0
    shortest_path_search_area = (40, 40, 40)

config = TestConfig()

class BrainbowGeneratorTest(unittest.TestCase):
    def test_detect_wrapping_neuron(self):
        gen = BrainbowGenerator(config)
        gen.image = np.zeros(config.image_size, dtype=np.uint16)
        gen.image[5:7, 5:7, :] = 1
        gen.neuron_buffer = np.ones(config.image_size, dtype=np.uint8)
        gen.dodge_locations = []

        gen.neuron_buffer[0:5, 5:7, 5:7] = 0
        gen.neuron_buffer[4:8, 2:5, 5:7] = 0
        gen.neuron_buffer[4:8, 7:9, 5:7] = 0
        gen.neuron_buffer[7:10, 5:7, 5:7] = 0

        gen.dodge_locations = [(np.array([4, 5, 5]), np.array([7, 5, 5]))]

        self.assertFalse(gen.insert_new_neuron())
        
