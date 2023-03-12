import unittest
import numpy as np

from config.brainbows.v_1_0_0 import config
from image_synthesis.brainbows.v_1_0 import BrainbowGenerator


class BrainbowGeneratorTest(unittest.TestCase):
    def test_invalid_new_neuron(self):
        generator = BrainbowGenerator(config)
        generator.neuron_count = 1
        image = np.array([[
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]],

            [[0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]],

            [[0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]]
        ])
        neuron = np.array([[
            [False, False, False],
            [False, False, False],
            [False, False, False]],

            [[False, False, False],
            [False, True, False],
            [False, False, False]],

            [[False, False, False],
            [False, False, False],
            [False, False, False]]
        ])
        self.assertFalse(generator.valid_new_neuron(image, neuron))

        image = np.zeros((10, 10, 10))
        image[0:2, 0, 0:2] = 1
        image[2, 0:2, 0:2] = 1
        image[3, 1:3, 0:2] = 1
        image[4, 2, 0:2] = 1

        neuron = np.zeros((10, 10, 10), dtype=bool)
        neuron[2, 0, 0:10] = True
        neuron[2, 1, 0:10] = True
        neuron[3, 1, 0:10] = True
        self.assertFalse(generator.valid_new_neuron(image, neuron))

        neuron = np.zeros((10, 10, 10), dtype=bool)
        neuron[2, 1, 0:10] = True
        self.assertFalse(generator.valid_new_neuron(image, neuron))

        neuron = np.ones((10, 10, 10), dtype=bool)
        self.assertFalse(generator.valid_new_neuron(image, neuron))

        generator.neuron_count = 2
        image = np.zeros((5, 5, 5))
        image[0, 0, 0:5] = 1
        image[0, 3, 0:5] = 2

        neuron = np.zeros((5, 5, 5), dtype=bool)
        neuron[0, 2:5, 3] = True
        self.assertFalse(generator.valid_new_neuron(image, neuron))

    def test_valid_new_neuron(self):
        generator = BrainbowGenerator(config)
        generator.neuron_count = 1
        image = np.array([[
            [0, 0, 0],
            [1, 1, 0],
            [0, 0, 0]],

            [[0, 0, 0],
            [1, 1, 0],
            [0, 0, 0]],

            [[0, 0, 0],
            [1, 1, 0],
            [0, 0, 0]]
        ])
        neuron = np.array([[
            [False, False, False],
            [False, False, False],
            [False, False, False]],

            [[False, False, False],
            [False, True, False],
            [False, False, False]],

            [[False, False, False],
            [False, False, False],
            [False, False, False]]
        ])
        self.assertTrue(generator.valid_new_neuron(image, neuron))

        image = np.zeros((10, 10, 10))
        image[0:2, 0, 0:2] = 1
        image[2, 0:2, 0:2] = 1
        image[3, 1:3, 0:2] = 1
        image[4, 2, 0:2] = 1

        neuron = np.zeros((10, 10, 10), dtype=bool)
        neuron[3, 1:3, 0:5] = True
        neuron[4, 2, 0:5] = True
        self.assertTrue(generator.valid_new_neuron(image, neuron))

        generator.neuron_count = 2
        image = np.zeros((10, 10, 10))
        image[1, :, :] = 1
        image[6, :, :] = 2

        neuron = np.zeros((10, 10, 10), dtype=bool)
        neuron[:, 2:8, 5] = True

        self.assertTrue(generator.valid_new_neuron(image, neuron))
