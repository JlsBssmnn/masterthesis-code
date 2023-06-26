import json
from skimage.metrics import variation_of_information
import webbrowser
import neuroglancer
import h5py
from tqdm import tqdm
from skimage.segmentation import watershed
import skimage
from pathlib import Path
import numpy as np
from evaluation.config.template import EpithelialSegmentationConfig
from evaluation.evaluation_utils import NpEncoder, get_path
from utils.neuroglancer_viewer.neuroglancer_viewer import show_image

class Evaluation:
    def __init__(self, under_segmentation, over_segmentation, segmentation, acc):
        self.under_segmentation = under_segmentation
        self.over_segmentation = over_segmentation
        self.variation_of_information = under_segmentation + over_segmentation
        self.segmentation = segmentation
        self.acc = acc

    def write_to_dict(self, d):
        """
        Stores the evalution to the given dict `d`.
        """
        d["under_segmentation"] = self.under_segmentation
        d["over_segmentation"] = self.over_segmentation
        d["variation_of_information"] = self.variation_of_information
        d["acc"] = self.acc

class SEEpithelial:
    def __init__(self, config: EpithelialSegmentationConfig, use_get_path=True):
        self.config = config
        self.membrane_black = self.config.membrane_black

        self.ground_truths = []
        self.ground_truth_masks = [] # stores masks for ground truths indicating which pixels actually can be considered
                                     # as truth (i.e. for membrane truth all membrane pixels, for cell truth all cell pixels)
        self.cell_truth_pixels = []  # The number of pixels that are cells in each cell truth
        datasets = self.config.ground_truth_datasets
        ground_truth_path = get_path(self.config.ground_truth_file) if use_get_path else self.config.ground_truth_file

        with h5py.File(ground_truth_path, 'r') as f:
            for dataset_list in datasets:
                assert len(dataset_list) == 2, 'For epithelial sheet, cell and membrane truth are neede'
                membrane_truth = np.asarray(f[dataset_list[0]])
                cell_truth = np.asarray(f[dataset_list[1]])
                self.ground_truths.append([membrane_truth, cell_truth])
                self.ground_truth_masks.append([membrane_truth == 0, cell_truth != 0])
                self.cell_truth_pixels.append((cell_truth != 0).sum())

        self.results = {
            "config": {
                "basins_range": self.config.basins_range,
                "membrane_range": self.config.membrane_range,
                "membrane_black": self.config.membrane_black,
                "error_factor": self.config.error_factor,
            },
            "evaluation": []
        }

        basin_thresholds = np.arange(*self.config.basins_range)
        membrane_thresholds = np.arange(*self.config.membrane_range)
        basin_thresholds, membrane_thresholds = np.meshgrid(basin_thresholds, membrane_thresholds)

        indices = np.stack((basin_thresholds.flatten(), membrane_thresholds.flatten()), axis=1)

        if config.membrane_black:
            self.threshold_values = indices[indices[:, 0] > indices[:, 1]]
        else:
            self.threshold_values = indices[indices[:, 0] < indices[:, 1]]
        assert self.threshold_values.shape[0] > 0

    def clear(self):
        self.results["evaluation"] = []

    def determine_file_path(self):
        """
        Looks in the output directory and determines an appropriate name for the output.
        """
        assert self.config.save_directory is not None
        path = Path(get_path(self.config.save_directory))
        assert path.is_dir()
        
        name = self.config.save_file_name
        duplicates = list(path.glob(name + '*'))

        if len(duplicates) == 0:
            return path / (name + '.json')

        version = 1
        for duplicate in duplicates:
            last_part = duplicate.stem.split('_')[-1]
            if not last_part.isdigit():
                continue
            version = max(int(last_part) + 1, version)
        file_name = f'{name}_{version}.json'

        return path / file_name

    def compute_score(self, evaluation):
        vi_part = self.config.error_factor * (evaluation.variation_of_information)

        local_error_measure = getattr(evaluation, self.config.local_error_measure)
        if local_error_measure <= self.config.local_error_a:
            local_error = ((1-self.config.local_error_b) / self.config.local_error_a - 1) * \
                local_error_measure + self.config.local_error_b
        else:
            local_error = 1 - local_error_measure
        local_error *= (1 - self.config.error_factor)
        return vi_part + local_error
            

    def find_segmentation_and_eval(self, images):
        """
        Find the best segmentation for the list of images. Image values should be in the range between 0 and 1.
        """
        assert len(images) == len(self.config.ground_truth_datasets)
        assert len(images) == len(self.config.image_names)

        prog = tqdm(total=self.threshold_values.shape[0] * len(images), disable=not self.config.show_progress)
        segmentations = {}
        images = list(map(lambda x: eval(f'x[{self.config.slice_str}]'), images))
        images_uint8 = list(map(lambda x: (x * 255).astype(np.uint8), images))

        for i, image in enumerate(images):
            membrane_truth, cell_truth = self.ground_truths[i]
            result = {
                "segmentation_parameters": (seg_params := {"tweak_image": self.config.image_names[i]}),
                "evaluation_scores": {k: dict() for k in self.config.image_names}
            }
            tweak_image_result = result["evaluation_scores"][self.config.image_names[i]]

            best_score = float('inf')
            evaluation = None
            for basin_threshold, membrane_threshold in self.threshold_values:
                evaluation = self.eval_image(images_uint8[i], basin_threshold, membrane_threshold, membrane_truth, cell_truth, i)

                score = self.compute_score(evaluation)
                if score < best_score:
                    best_score = score
                    evaluation.write_to_dict(tweak_image_result)
                    tweak_image_result["score"] = score
                    seg_params["basin_threshold"] = basin_threshold
                    seg_params["membrane_threshold"] = membrane_threshold
                    segmentations[f"slice{i}"] = evaluation.segmentation
                prog.update()

            diff = self.compute_diff(image, i)
            tweak_image_result["diff"] = diff

            if best_score == float('inf'):
                self.results["evaluation"].append(result) 
                continue

            basin_threshold = seg_params["basin_threshold"]
            membrane_threshold = seg_params["membrane_threshold"]

            for j in range(len(images)):
                if i == j:
                    continue

                other_image_result = result["evaluation_scores"][self.config.image_names[j]]
                membrane_truth, cell_truth = self.ground_truths[j]

                evaluation = self.eval_image(images_uint8[j], basin_threshold, membrane_threshold, membrane_truth, cell_truth, j)
                score = self.compute_score(evaluation)
                evaluation.write_to_dict(other_image_result)
                other_image_result["score"] = score
            self.results["evaluation"].append(result) 
        prog.close()
        return segmentations

    def eval_and_store(self, images):
        segmentations = self.find_segmentation_and_eval(images)

        json_string = json.dumps(self.results, indent=4, cls=NpEncoder)
        if self.config.save_results:
            with open(self.determine_file_path(), 'x') as f:
                f.write(json_string)
        else:
            print(json_string)

        if self.config.show_segmentation:
            viewer = neuroglancer.Viewer()
            for name, image in segmentations.items():
                show_image(viewer, image, name=name, segmentation=not name.endswith('prec_calc'))
            webbrowser.open(str(viewer), new=0, autoraise=True)
            input("Done?")

    def eval_image(self, image_uint8, basin_threshold, membrane_threshold, membrane_truth, cell_truth, i) -> Evaluation:
        # compute segmentation
        labels = skimage.morphology.label(image_uint8 >= basin_threshold if self.membrane_black else image_uint8 <= basin_threshold,
                                          connectivity=1)
        segmentation =  watershed(-image_uint8 if self.membrane_black else image_uint8, markers=labels,
                         mask=image_uint8 > membrane_threshold if self.membrane_black else image_uint8 < membrane_threshold)

        # compute VI
        segmentation_cells = (segmentation != 0)
        membrane_mask = segmentation_cells & (membrane_truth != 0)
        cell_mask = segmentation_cells & self.ground_truth_masks[i][1]

        if membrane_mask.any():
            _, under_segmentation = variation_of_information(membrane_truth[membrane_mask], segmentation[membrane_mask], ignore_labels=[0])
        else:
            under_segmentation = float('nan')

        if cell_mask.any():
            over_segmentation, _ = variation_of_information(cell_truth[cell_mask], segmentation[cell_mask], ignore_labels=[0])
        else:
            over_segmentation = float('nan')

        # compute acc
        pred_cells = segmentation[self.ground_truth_masks[i][1]] != 0
        acc = pred_cells.sum() / self.cell_truth_pixels[i]

        return Evaluation(under_segmentation, over_segmentation, segmentation, acc)

    def compute_diff(self, image, i):
        membrane_diff = image[self.ground_truth_masks[i][0]] - (not self.membrane_black)
        cell_diff = image[self.ground_truth_masks[i][1]] - self.membrane_black
        return (np.append(cell_diff, membrane_diff) ** 2).mean()

def find_segmentation_and_eval(images, config: EpithelialSegmentationConfig):
    evaluater = SEEpithelial(config)
    return evaluater.eval_and_store(images)

def one_step(config: EpithelialSegmentationConfig):
    """
    Can be used to start segmentation from an image file. The result is saved to an output file.
    """
    assert config.input_file is not None
    assert config.input_datasets is not None
    assert 0 < len(config.input_datasets)

    images = []
    with h5py.File(config.input_file) as f:
        for dataset in config.input_datasets:
            images.append(np.asarray(f[dataset]))

    find_segmentation_and_eval(images, config)
