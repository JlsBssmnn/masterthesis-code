from typing import Any

class TranslateImageConfig:
    batch_size: int                  # How many input patches are fed into the generator at once
    generator_config: Any            # The parameters that the generator was created with (as another object)
    generator_save: str              # Path to the saved generator
    input_dataset: str               # The dataset which contains the input image
    input_file: str                  # The file that contains the input image
    patch_size: tuple[int, int, int] # The input size for the generator
    scale_with_patch_max: bool       # If true, the network input is computed like this: (patch / patch.max() - 0.5) * 2
    slices: list[str] | None         # Python slice strings. If provided only these parts of the input are fed to the generator.
    use_gpu: bool                    # Whether to do the inference on the gpu

    # these properties are only used in the one_step function
    output_datasets: list[str] | None # The dataset names of the output images
    output_file: str | None           # Path to the output file (where result is written to)
    save_images: bool                 # If true, the resulting images are saved
    show_images: bool                 # If true, the results are shown in neuroglancer

class SegmentationConfig:
    basins_range: tuple[float, float, float]     # start, stop and step values for searching for basin threshold parameters
    error_factor: float                          # The factor that is used to combine global and local error
    ground_truth_datasets: list[tuple[str, str]] # The datasets in the ground truth file that contain the images. The first dataset is
                                                 # used for determining the undersegmentation, second for oversegmentation
    ground_truth_file: str                       # The file containing the ground truth
    image_names: list[str]                       # Names of images. This will be used to store evaluation metrics for the images
    local_error_a: float                         # The a parameter for computing the local error
    local_error_b: float                         # The b parameter for computing the local error
    local_error_measure: str                     # Name of the measure that is used for the local error
    membrane_black: bool                         # True if the membrane color is black, False if it's white
    membrane_range: tuple[float, float, float]   # start, stop and step values for searching for membrane threshold parameters
    save_directory: str                          # The directory where to save the results
    save_file_name: str                          # The name of the saved file
    save_results: bool                           # If true, save results to a file, else just print results
    show_progress: bool                          # If true, a progress bar will indicate the progress
    show_segmentation: bool                      # If true, the segmentation images will be shown
    slice_str: str                               # A python slice specifying which part of the generator output is used for segmentation

    # these properties are only used in the one_step function
    input_file: str | None            # The file for loading the images
    input_datasets: list[str] | None  # The datasets in the input file which shall be used

class Config:
    translate_image_config: TranslateImageConfig
    segmentation_config: SegmentationConfig

config = TranslateImageConfig()
