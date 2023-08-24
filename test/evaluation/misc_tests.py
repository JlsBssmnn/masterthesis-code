import unittest
import numpy as np
import numpy.testing as npt
from itertools import product

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent / 'cycleGAN'))

from evaluation.translate_image import GeneratorApplier
# from evaluation.evaluate_segmentation import evaluate_segmentation_epithelial

class Config:
    use_gpu = False

    def __init__(self, patch_size):
        self.patch_size = patch_size

class ApplyGeneratorTest(unittest.TestCase):
    def test_compute_patch_locations1(self):
        shape = (1, 32, 64, 64)
        patch_size = np.array([32, 64, 64])
        applier = GeneratorApplier(shape, Config(patch_size))

        patch_locations = applier.compute_patch_locations(shape, patch_size)
        npt.assert_equal(patch_locations[0], np.array([[0, 0, 0]]))
        npt.assert_equal(patch_locations[1], np.empty((0, 3)))
        npt.assert_equal(patch_locations[2], np.empty((0, 3)))
        npt.assert_equal(patch_locations[3], np.empty((0, 3)))

        patch_size = np.array([16, 32, 32])

        patch_locations = applier.compute_patch_locations(shape, patch_size)
        npt.assert_equal(patch_locations[0], np.array([[0, 0, 0], [0, 0, 32], [0, 32, 0],
            [0, 32, 32], [16, 0, 0], [16, 0, 32], [16, 32, 0], [16, 32, 32]]))
        npt.assert_equal(patch_locations[1], np.array([[8, 0, 0], [8, 0, 32], [8, 32, 0], [8, 32, 32]]))
        npt.assert_equal(patch_locations[2], np.array([[0, 16, 0], [0, 16, 32], [16, 16, 0], [16, 16, 32]]))
        npt.assert_equal(patch_locations[3], np.array([[0, 0, 16], [0, 32, 16], [16, 0, 16], [16, 32, 16]]))

class EvaluationTest(unittest.TestCase):
    def test_evaluate_segmentation_epithelial(self):
        membrane_truth = np.array([
            [1, 1, 1, 1, 0],
            [1, 1, 1, 0, 3],
            [0, 0, 0, 3, 3],
            [2, 2, 2, 0, 3],
            [2, 2, 2, 2, 0],
        ])
        cell_truth = np.array([
            [1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 3],
            [2, 0, 0, 0, 0],
            [2, 2, 2, 2, 0],
        ])
        segmentation = np.array([
            [1, 1, 1, 1, 0],
            [1, 0, 1, 0, 3],
            [0, 0, 0, 0, 3],
            [0, 2, 2, 0, 3],
            [2, 2, 2, 0, 3],
        ])

        # All outputs differ only at locations where cell and membrane truth contradict
        # l1 diffs to cell: 0.1, 0, 0.1, 0.2, 0.2, 0, 0, 0.2
        # l1 diffs to membrane: 0.1, 0, 0, 0.2, 0.1, 0, 0.3
        net_output1 = np.array([
            [0.9, 1.0, 1.0, 0.5, 0.1],
            [0.2, 0.3, 1.0, 0.0, 0.9],
            [0.0, 0.2, 0.1, 0.0, 0.9],
            [1.2, 0.2, 0.0, 0.0, 0.1],
            [0.8, 1.0, 1.0, 0.8, 0.3],
        ])
        net_output2 = np.array([
            [0.9, 1. , 1. , 1. , 0.1],
            [1. , 1. , 1. , 0. , 1. ],
            [0. , 0.2, 0.1, 1. , 0.9],
            [1.2, 1. , 1. , 0. , 1. ],
            [0.8, 1. , 1. , 0.8, 0.3]
        ])
        net_output3 = np.array([
            [0.9, 1. , 0. , 0. , 0.1],
            [0. , 0. , 0. , 0. , 0. ],
            [0. , 0.2, 0.1, 0. , 0.9],
            [1.2, 0. , 0. , 0. , 0. ],
            [0.8, 1. , 1. , 0.8, 0.3]
        ])

        e1 = evaluate_segmentation_epithelial(net_output1, segmentation, membrane_truth, cell_truth, True)
        e2 = evaluate_segmentation_epithelial(net_output2, segmentation, membrane_truth, cell_truth, True)
        e3 = evaluate_segmentation_epithelial(net_output3, segmentation, membrane_truth, cell_truth, True)

        self.assertAlmostEqual(e1.diff, 0.019333333333333334)
        self.assertAlmostEqual(e2.diff, 0.019333333333333334)
        self.assertAlmostEqual(e3.diff, 0.019333333333333334)
