import unittest
import numpy as np

from image_synthesis.brainbows.v_1_0 import BrainbowGenerator
from image_synthesis.logging_config import logging

logging.getLogger().setLevel(logging.ERROR)

class TestConfig:
  image_size = (10, 10, 10)
  max_branches = 2
  max_thickness = 20
  min_thickness = 2
  noise_resolution_factor = 1
  perlin_max = 0.7
  seed = 0

config = TestConfig()

class BrainbowGeneratorTest(unittest.TestCase):
    def test_splitting_neuron(self):
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
        generator.image = image
        self.assertTrue(generator.insert_new_neuron(neuron))
        self.assertIn(generator.image[0, 1, 1], [0, 1])
        self.assertIn(generator.image[2, 1, 1], [0, 1])
        self.assertNotEqual(generator.image[0, 1, 1], generator.image[2, 1, 1])
        self.assertEqual(generator.image[1, 1, 1], 2)

        image = np.zeros((10, 10, 10), dtype=np.uint16)
        image[0:2, 0, 0:2] = 1
        image[2, 0:2, 0:2] = 1
        image[3, 1:3, 1:6] = 1
        image[4, 2, 1:4] = 1

        neuron = np.zeros((10, 10, 10), dtype=bool)
        neuron[2, 0:2, 0:10] = True
        neuron[3, 1, 0:10] = True
        generator.image = image.copy()
        generator.neuron_count = 1

        self.assertTrue(generator.insert_new_neuron(neuron))
        self.assertEqual(np.unique(generator.image[0:2, 0, 0:2])[0], 1)
        self.assertEqual((generator.image == 1).sum(), 4)
        self.assertEqual(generator.image[2, 0, 0], 2)
        self.assertEqual(generator.image[4, 2, 1], 0)

        neuron = np.zeros((10, 10, 10), dtype=bool)
        neuron[2, 1, 0:10] = True

        generator.image = image.copy()
        generator.neuron_count = 1
        self.assertTrue(generator.insert_new_neuron(neuron))

        image = np.zeros((5, 5, 5), dtype=np.uint16)
        image[0, 0, 0:5] = 1
        image[0, 3, 0:5] = 2

        neuron = np.zeros((5, 5, 5), dtype=bool)
        neuron[0, 2:5, 3] = True

        generator.image = image
        generator.neuron_count = 2

        self.assertTrue(generator.insert_new_neuron(neuron))
        self.assertIn(generator.image[0, 3, 0], [0, 2])
        self.assertIn(generator.image[0, 3, 4], [0, 2])
        self.assertNotEqual(generator.image[0, 3, 0], generator.image[0, 3, 4])
        self.assertEqual(generator.image[0, 3, 3], 3)

    def test_invalid_neuron(self):
        generator = BrainbowGenerator(config)
        image = np.zeros((10, 10, 10), dtype=np.uint16)
        image[0:2, 0, 0:2] = 1
        image[2, 0:2, 0:2] = 1
        image[3, 1:3, 1:6] = 1
        image[4, 2, 1:4] = 1

        generator.image = image.copy()
        generator.neuron_count = 1

        neuron = np.ones((10, 10, 10), dtype=bool)
        self.assertFalse(generator.insert_new_neuron(neuron))
        self.assertEqual(generator.neuron_count, 1)

        image = np.zeros((10, 10, 10), dtype=np.uint16)
        image[:, 3:8, 2:5] = 1

        neuron = np.zeros((10, 10, 10), dtype=bool)
        neuron[0:2, :, :] = True
        neuron[:, 2, :] = True
        neuron[5, 2:8, 2:6] = True
        neuron[9, :, :] = True

        generator.image = image.copy()
        generator.neuron_count = 1

        self.assertFalse(generator.insert_new_neuron(neuron))
        self.assertEqual(generator.neuron_count, 1)

    def test_non_splitting_neuron(self):
        generator = BrainbowGenerator(config)
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

        generator.image = image
        generator.neuron_count = 1

        self.assertTrue(generator.insert_new_neuron(neuron))
        self.assertTrue((generator.image == np.array([
          [[0, 0, 0],
           [1, 1, 0],
           [0, 0, 0]],
          [[0, 0, 0],
           [1, 2, 0],
           [0, 0, 0]],
          [[0, 0, 0],
           [1, 1, 0],
           [0, 0, 0]]
          ])).all())

        image = np.zeros((10, 10, 10), dtype=np.uint16)
        image[0:2, 0, 0:2] = 1
        image[2, 0:2, 0:2] = 1
        image[3, 1:3, 0:2] = 1
        image[4, 2, 0:2] = 1

        neuron = np.zeros((10, 10, 10), dtype=bool)
        neuron[3, 1:3, 0:5] = True
        neuron[4, 2, 0:5] = True

        generator.image = image
        generator.neuron_count = 1

        self.assertTrue(generator.insert_new_neuron(neuron))
        self.assertEqual(generator.image[2, 1, 1], 1)
        self.assertEqual(generator.image[3, 2, 1], 2)
        self.assertEqual(generator.neuron_count, 2)

        image = np.zeros((10, 10, 10), dtype=np.uint16)
        image[1, :, :] = 1
        image[6, :, :] = 2

        neuron = np.zeros((10, 10, 10), dtype=bool)
        neuron[:, 2:8, 5] = True

        generator.image = image
        generator.neuron_count = 2

        self.assertTrue(generator.insert_new_neuron(neuron))
        self.assertEqual(generator.neuron_count, 3)
