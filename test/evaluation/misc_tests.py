import unittest
import numpy as np
import numpy.testing as npt
from itertools import product

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent / 'cycleGAN'))

from evaluation.translate_image import compute_patch_locations
from evaluation.evaluate_segmentation import evaluate_segmentation_epithelial

class ApplyGeneratorTest(unittest.TestCase):
    def test_compute_patch_locations1(self):
        shape = (1, 32, 64, 64)
        patch_size = np.array([32, 64, 64])
        stride = np.array([32, 64, 64])

        patch_locations = compute_patch_locations(shape, patch_size, stride)
        npt.assert_equal(patch_locations, np.array([[0, 0, 0]]))

        stride = np.array([16, 32, 32])
        patch_locations = compute_patch_locations(shape, patch_size, stride)
        npt.assert_equal(patch_locations, np.array([[0, 0, 0]]))
        
        stride = np.array([8, 16, 16])
        patch_locations = compute_patch_locations(shape, patch_size, stride)
        npt.assert_equal(patch_locations, np.array([[0, 0, 0]]))

        patch_size = np.array([16, 32, 32])
        stride = np.array([16, 32, 32])

        patch_locations = compute_patch_locations(shape, patch_size, stride)
        npt.assert_equal(patch_locations, np.array([[0, 0, 0], [0, 0, 32], [0, 32, 0],
            [0, 32, 32], [16, 0, 0], [16, 0, 32], [16, 32, 0], [16, 32, 32]]))

        stride = np.array([10, 12, 32])
        patch_locations = compute_patch_locations(shape, patch_size, stride)
        npt.assert_equal(patch_locations, np.array(list(product([0, 10, 16], [0, 12, 24, 32], [0, 32]))))

    def test_compute_patch_locations2(self):
        shape = (1, 1, 4, 10)
        patch_size = np.array([1, 4, 4])
        stride = np.array([1, 2, 2])

        patch_locations = compute_patch_locations(shape, patch_size, stride)
        npt.assert_equal(patch_locations, np.array([[0, 0, 0], [0, 0, 2], [0, 0, 4], [0, 0, 6]]))

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
