import argparse
import numpy as np
from skimage import color
import sys

from utils.segment_colors import get_colors

class StoreDictKeyPair(argparse.Action):
     def __init__(self, option_strings, dest, nargs=None, **kwargs):
         self._nargs = nargs
         super(StoreDictKeyPair, self).__init__(option_strings, dest, nargs=nargs, **kwargs)
     def __call__(self, parser, namespace, values, option_string=None):
         my_dict = {}
         for kv in values:
             k,v = kv.split("=")
             my_dict[k] = v
         setattr(namespace, self.dest, my_dict)

def scale_image(image, in_min, in_max, out_min, out_max):
    assert in_min < in_max and out_min < out_max

    in_diff = abs(in_max - in_min)
    out_diff = abs(out_max - out_min)
    in_min = in_min
    in_max = in_max
    scaling = out_diff / in_diff
    offset = out_min - in_min * scaling

    return image * scaling + offset

def label2rgb(array, seed=0):
    return color.label2rgb(array, colors=get_colors(seed))
