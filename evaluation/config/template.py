from typing import Any, Literal
import numpy.typing as npt

class TranslateImageConfig:
    batch_size: int                     # How many input patches are fed into the generator at once
    generator_config: Any               # The parameters that the generator was created with (as another object)
    generator_save: str                 # Path to the saved generator
    input_dataset: str                  # The dataset which contains the input image
    input_file: str                     # The file that contains the input image
    mask: None | npt.NDArray[bool]      # If provided, the mask is applied to generator output and only the masked output is used
    masked_value: float                 # The value assigned to voxels outside of the mask for the generator output
    patch_size: tuple[int, int, int]    # The input size for the generator
    scale_with_patch_max: bool          # If true, the network input is computed like this: (patch / patch.max() - 0.5) * 2
    skip_translation: bool              # If true, translation is skipped. Can be used to evaluate without applying the generator.
    slices: list[str] | None            # Python slice strings. If provided only these parts of the input are fed to the generator.
    stride: tuple[int, int, int] | None # The stride for iterating over the images. Only required when using the old translation script.
    use_gpu: bool                       # Whether to do the inference on the gpu

    # these properties are only used in the one_step function
    output_datasets: list[str] | None # The dataset names of the output images
    output_file: str | None           # Path to the output file (where result is written to)
    save_images: bool                 # If true, the resulting images are saved
    show_images: bool                 # If true, the results are shown in neuroglancer

class EpithelialSegmentationConfig:
    basins_range: tuple[float, float, float]     # start, stop and step values for searching for basin threshold parameters
    error_factor: float                          # The factor that is used to combine global and local error
    ground_truth_datasets: list[tuple[str, str]] # The datasets in the ground truth file that contain the images. The first dataset is
                                                 # used for determining the undersegmentation, second for oversegmentation
    ground_truth_file: str                       # The file containing the ground truth
    image_names: list[str]                       # Names of images. This will be used to store evaluation metrics for the images
    local_error_a: float                         # The a parameter for computing the local error
    local_error_b: float                         # The b parameter for computing the local error
    local_error_measure: str                     # Name of the measure that is used for the local error (use 'acc' to use the recall)
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

class BrainbowSegmentationConfig:
    bg_measure: str | list[str]                        # Which function to use to convert color values to background probabilities
    bg_threshold: float | list[float]                  # The maximum value that identifies a background voxel
    bg_vi_weight: float                                # In [0, 1]; The ratio of the sum of background weights to the sum of all weights (for VI)
    bias_cut_range: tuple[float, float, float]         # The range of bias cuts that are searched
    dist_measure: str | list[str]                      # Which distance measure to use for convertin a color image to affinites
    ground_truth_dataset: str                          # The dataset that contains the ground truth
    ground_truth_file: str                             # The file containing the ground truth
    ground_truth_slices: list[str]                     # Slices in the ground truth image that shall be used
    image_names: list[str]                             # Names of images. This will be used to store evaluation metrics for the images
    image_type: Literal['color'] | Literal['affinity'] # Whether the image contains color values or affinity values
    mask_datasets: list[str | None] | None             # The mask datasets for each slice. List value of None means no mask
    mask_file: str | None                              # A file that contains masks for the slices. If none, masks won't be used
    offsets: list[tuple[int, int, int]]                # The offsets that are used in the mutex watershed algorithm
    save_directory: str                                # The directory where to save the results
    save_file_name: str                                # The name of the saved file
    save_results: bool                                 # If true, save results to a file, else just print results
    seperating_channel: int                            # Parameter for the mutex watershed algorithm
    show_progress: bool                                # If true, a progress bar will indicate the progress
    show_segmentation: bool                            # If true, the segmentation images will be shown
    slice_str: str                                     # A python slice specifying which part of the generator output is used for segmentation
    verbose: bool                                      # If true, logs will be emitted informing about the sate of the program

    # these properties are only used in the one_step function
    input_file: str | None            # The file for loading the images
    input_datasets: list[str] | None  # The datasets in the input file which shall be used

class Config:
    translate_image_config: TranslateImageConfig
    segmentation_config: EpithelialSegmentationConfig | BrainbowSegmentationConfig

config = Config()
