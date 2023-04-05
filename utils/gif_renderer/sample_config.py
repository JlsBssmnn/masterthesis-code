"""
This file shows a sample config. It doesn't contain actual values,
but instead just shows which datatypes are used. The different parameters
are also explained with comments.
Real configs must live in the ./configs directory and must provide an instance
of the config class which is named 'config'.
"""

from typing import Any


class Config:
    add_reverse: bool                  # If true, the reversed order of frames is added to the gif to enable smooth loops
    directories: list[str]             # A list of datasets and groups in the h5 file that are used to create the gifs
                                       # If a group is provided, all datasets within this group (not recursively) will be used
    duration: int                      # The duration of one gif frame in milliseconds
    filters: dict[str, dict[str, Any]] # The filters that are applied to the images and their parameters. The keys are the names of the filters,
                                       # the values are dictionaries where the keys are the parameter names and the values the parameter values
    input_file: str                    # The file that contains the 3D images that shall be converted to gifs
    loop: int                          # How often the gif loops, 0 means infinitely
    names: list[str]                   # The names of the output gif images. If there are less names than output images, the
                                       # default naming is applied after the name list is depleted which may overwrite created gifs
    output_dir: str                    # The directory where the gif files are stored
    slice_axis: int                    # The axis in the image that the gif is moving along over the frames

config = Config()
