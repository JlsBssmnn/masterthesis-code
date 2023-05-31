from typing import Any

class TranslateImageConfig:
    input_file: str                  # The file that contains the input image
    generator_save: str              # Path to the saved generator
    generator_config: Any            # The parameters that the generator was created with (as another object)
    slices: list[str] | None         # Python slice strings. If provided only these parts of the input are fed to the generator.
    input_dataset: str               # The dataset which contains the input image
    patch_size: tuple[int, int, int] # The input size for the generator
    stride: tuple[int, int, int]     # The stride for moving the through the input image
    batch_size: int                  # How many input patches are fed into the generator at once
    use_gpu: bool                    # Whether to do the inference on the gpu

    # these properties are only used in the one_step function
    output_datasets: list[str] | None # The dataset names of the output images
    output_file: str | None           # Path to the output file (where result is written to)

class SegmentationConfig:
    slice_str: str       # A python slice specifying which part of the generator output is used for segmentation
    tweak_image_idx: int # The index of the image that is used for tweaking the settings
    membrane_black: bool # True if the membrane color is black, False if it's white

    # these properties are only used in the one_step function
    input_file: str | None            # The file for loading the images
    input_datasets: list[str] | None  # The datasets in the input file which shall be used
    output_file: str | None           # The file for saving the output
    output_datasets: list[str] | None # The datasets for saving the output
    config_output_file: str | None    # The file for storing the chosen segmentation config

class EvaluationConfig:
    ground_truth_file: str                 # The file containing the ground truth
    ground_truth_datasets: list[list[str]] # The datasets in the ground truth file that contain the images. Each
                                           # list[str] can specify multiple ground truths for one image
    save_directory: str | None             # Where to save the results. The file name will be inferred. If None don't save.

    # these properties are only used in the one_step function
    segmentation_file: str | None           # The file that contains the segmentation result
    segmentation_datasets: list[str] | None # The datasets containing the segmentation images
    config_file: str | None                 # The file that stores the config that was used to achieve the segmentation

class Config:
    translate_image_config: TranslateImageConfig
    segmentation_config: SegmentationConfig
    evaluation_config: EvaluationConfig

config = TranslateImageConfig()
