import unittest

from image_synthesis.brainbows.colorize_brainbows import extract_subbits

class ColorizeBrainbowsTest(unittest.TestCase):
    def test_extract_subbits(self):
        self.assertRaises(AssertionError, lambda: extract_subbits(123, 1, 0))
        self.assertRaises(AssertionError, lambda: extract_subbits(123, 3, 1))
        self.assertRaises(AssertionError, lambda: extract_subbits(123, 1, 1))
        self.assertRaises(AssertionError, lambda: extract_subbits(123, -1, 3))
        self.assertRaises(AssertionError, lambda: extract_subbits(123, 1, -3))

        self.assertEqual(extract_subbits(2**32 - 1, 0, 8), 255)
        self.assertEqual(extract_subbits(2**32 - 1, 8, 16), 255)
        self.assertEqual(extract_subbits(2**32 - 1, 16, 24), 255)
        self.assertEqual(extract_subbits(2**32 - 1, 24, 32), 255)

        self.assertEqual(extract_subbits(37, 2, 5), 2)
        self.assertEqual(extract_subbits(37, 3, 6), 5)
        self.assertEqual(extract_subbits(37, 0, 6), 37)
        
        self.assertRaises(TypeError, lambda: extract_subbits(0, 8, 16))
        self.assertEqual(extract_subbits(0, 8, 16, 32), 0)
        self.assertEqual(extract_subbits(19, 8, 16, 32), 0)
        self.assertEqual(extract_subbits(19, 16, 32, 32), 19)
        self.assertEqual(extract_subbits(26, 2, 5), 2)
        self.assertEqual(extract_subbits(26, 2, 5, 8), 3)
