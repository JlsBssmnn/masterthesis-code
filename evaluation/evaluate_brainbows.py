import json
from tqdm import tqdm
from pathlib import Path
from cycleGAN.data.dynamic_dataset_creation import create_brainbow_affinities
from evaluation.evaluation_utils import NpEncoder, get_path
from evaluation.config.template import BrainbowSegmentationConfig
import numpy as np
from mutex_watershed import MutexWatershed
import skimage
import h5py
import neuroglancer
from utils.neuroglancer_viewer.neuroglancer_viewer import show_image
import webbrowser
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from partition_comparison import WeightedSingletonVariationOfInformation as WeightedSingletonVI, VariationOfInformation

# Function was copied from here: https://github.com/markschoene/MeLeCoLe/blob/main/melecole/infer.py#L181
def run_mws(affinities,
            offsets, stride,
            foreground_mask,
            seperating_channel=3,
            invert_dam_channels=True,
            bias_cut=0.,
            randomize_bounds=True,):
    """
    This code was copied from https://github.com/hci-unihd/mutex-watershed/
    """

    assert len(affinities) == len(offsets), "%s, %i" % (str(affinities.shape), len(offsets))
    affinities_ = np.require(affinities.copy(), requirements='C')
    if invert_dam_channels:
        affinities_[seperating_channel:] *= -1
        affinities_[seperating_channel:] += 1
    affinities_[:seperating_channel] += bias_cut

    # sort in descending order
    sorted_edges = np.argsort(-affinities_.ravel())

    # remove edges adjacent to background voxels from graph
    sorted_edges = sorted_edges[foreground_mask.ravel()[sorted_edges]]

    # run the mutex watershed
    vol_shape = affinities_.shape[1:]
    mst = MutexWatershed(np.array(vol_shape),
                         offsets,
                         seperating_channel,
                         stride)
    if randomize_bounds:
        mst.compute_randomized_bounds()
    mst.repulsive_ucc_mst_cut(sorted_edges, 0)
    segmentation = mst.get_flat_label_image_only_merged_pixels().reshape(vol_shape)
    return segmentation

# Function was copied from here: https://github.com/markschoene/MeLeCoLe/blob/main/melecole/infer.py#L220
def get_foreground_mask(affinities, membranes, offsets, theta, exclude_exterior=False):
    """
    Compute the foreground affinities given membrane predictions and a threshold.
    :param affinities: Metric affinities
    :param membranes: Foreground prediction (where 1 is foreground and 0 is background)
    :param offsets: offset vectors defining the affinity map
    :param theta: Threshold
    :param exclude_exterior: If true, all connections where the target is beyond image boundaries are set to false
    :return: Foreground mask (where 1 is foreground and 0 is background) that removes all background voxels and their corresponding affinities
    """
    mask = np.ones_like(affinities).astype(bool)

    # get largest offset number
    pad_size = np.max(np.abs(np.array(offsets)))

    # initialize padded foreground
    if exclude_exterior:
        foregound = np.pad(membranes > theta, pad_width=pad_size, mode='constant', constant_values=0).astype(bool)
    else:
        foregound = np.pad(membranes > theta, pad_width=pad_size, mode='constant', constant_values=1).astype(bool)

    # compute foreground mask for each offset vector
    for i, vector in enumerate(offsets):
        dims = membranes.shape
        slices_null = [slice(pad_size, pad_size + dims[k]) for k in range(len(dims))]
        slices_plus = [slice(pad_size + vector[k], pad_size + vector[k] + dims[k]) for k in range(len(dims))]

        # remove both edges that are associated with pixel (i, j, k)
        # that is (offset_1, offset_2, offset_3) + (i, j, k) AND (i, j, k)
        mask[i] = np.logical_and(foregound[slices_plus[0], slices_plus[1], slices_plus[2]],
                                 foregound[slices_null[0], slices_null[1], slices_null[2]])

    return mask

def segmentation_to_affinities(image, offsets, bg_value=0):
    """
    Converts a segmentation (z, y, x)-array to the affinity representation.
    """
    assert image.ndim == 3
    affinities = np.empty((len(offsets) + 1, *image.shape), dtype=np.float32)
    affinities[0] = image != bg_value

    for i, offset in enumerate(offsets):
        rolled_image = np.roll(image, -offset, (-3, -2, -1))
        affinities[i + 1] = image == rolled_image
    return affinities

class Evaluation:
    def __init__(self, vi, segmentation):
        self.under_segmentation = vi.valueFalseJoin()
        self.over_segmentation = vi.valueFalseCut()
        self.weighted_VI = vi.value()
        self.segmentation = segmentation

    @classmethod
    def create(cls, under_segmentation, over_segmentation, segmentation):
        obj = cls.__new__(cls)
        obj.under_segmentation = under_segmentation
        obj.over_segmentation = over_segmentation
        obj.weighted_VI = under_segmentation + over_segmentation
        obj.segmentation = segmentation

        return obj

    def write_to_dict(self, d):
        """
        Stores the evaluation to the given dict `d`.
        """
        d["weighted_under_seg"] = self.under_segmentation
        d["weighted_over_seg"] = self.over_segmentation
        d["weighted_VI"] = self.weighted_VI

class SEBrainbow:
    """
    A class that Segments brainbow images and Evaluates them.
    """
    def __init__(self, config: BrainbowSegmentationConfig):
        self.config = config
        self.config.offsets = np.array(self.config.offsets)
        self.color_image = config.image_type == 'color'
        self.bias_cut_values = np.arange(*self.config.bias_cut_range)

        self.ground_truths = []
        with h5py.File(config.ground_truth_file) as f:
            h5_dataset = f[config.ground_truth_dataset]
            for s in config.ground_truth_slices:
                self.ground_truths.append(np.asarray(eval(f"h5_dataset[{s}]")))

        self.ground_truth_affinites = [segmentation_to_affinities(image, self.config.offsets)
                                       for image in self.ground_truths]
        self.ground_truth_foregrounds = [get_foreground_mask(gt[1:], gt[0], self.config.offsets, 0.5)
                                         for gt in self.ground_truth_affinites]
        self.vi_weights = self.compute_vi_weights()

        self.results = {
            "config": {
                "bias_cut_range": self.config.bias_cut_range,
                "offsets": self.config.offsets,
                "bg_threshold": self.config.bg_threshold,
                "image_type": self.config.image_type,
                "seperating_channel": self.config.seperating_channel,
                "bg_vi_weight": self.config.bg_vi_weight,
            },
            "evaluation": []
        }

        if self.config.image_type == 'color':
            self.results['config'] |= {
                "bg_measure": self.config.bg_measure,
                "dist_measure": self.config.dist_measure,
            }

    def clear(self):
        self.results["evaluation"] = []

    def compute_vi_weights(self):
        weight_list = []
        for image in self.ground_truths:
            num_zero = np.count_nonzero(image == 0)
            num_non_zero = image.size - num_zero
            zero_weight = (num_non_zero * self.config.bg_vi_weight) / (num_zero * (1 - self.config.bg_vi_weight))
            weights = np.ones(image.shape)
            weights[image == 0] = zero_weight

            weight_list.append(weights)

        return weight_list

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

    def find_segmentation_and_eval(self, images, compute_VI=True):
        images = list(map(lambda x: eval(f'x[{self.config.slice_str}]'), images))
        if self.color_image:
            images = [create_brainbow_affinities(image, self.config.offsets, self.config.bg_measure, None, self.config.dist_measure)
                      for image in images]
        foreground_masks = [get_foreground_mask(image[1:], image[0], self.config.offsets, self.config.bg_threshold)
                            for image in images]

        used_bias_cuts = {}
        segmentations = {}
        prog = tqdm(total=self.bias_cut_values.shape[0] * len(images), disable=not self.config.show_progress)

        for i, image in enumerate(images):
            image_name = self.config.image_names[i]
            affinities = image[1:]
            foreground_mask = foreground_masks[i]
            result = {
                "segmentation_parameters": (seg_params := {"tweak_image": image_name}),
                "evaluation_scores": {k: dict() for k in self.config.image_names}
            }
            tweak_image_result = result["evaluation_scores"][image_name]

            self.compute_foreground_metrics(image, i, tweak_image_result)
            self.compute_affinity_metrics(image, i, tweak_image_result)

            if not compute_VI:
                self.results["evaluation"].append(result) 
                continue

            best_score = float('inf')
            cached_seg = segmentations[image_name] if image_name in segmentations else None

            # find optimal segmentation
            for bias_cut in np.arange(*self.config.bias_cut_range):
                if bias_cut in used_bias_cuts:
                    evaluation = self.results["evaluation"][used_bias_cuts[bias_cut]]['evaluation_scores'][image_name]
                    evaluation = Evaluation.create(evaluation['weighted_under_seg'], evaluation['weighted_over_seg'], cached_seg)

                    if evaluation.weighted_VI < best_score:
                        best_score = evaluation.weighted_VI
                        evaluation.write_to_dict(tweak_image_result)
                        seg_params["bias_cut"] = bias_cut
                        segmentations[image_name] = cached_seg
                    prog.update()
                    continue
                evaluation = self.eval_image(affinities, foreground_mask, bias_cut, i)

                if evaluation.weighted_VI < best_score:
                    best_score = evaluation.weighted_VI
                    evaluation.write_to_dict(tweak_image_result)
                    seg_params["bias_cut"] = bias_cut
                    segmentations[image_name] = evaluation.segmentation
                prog.update()


            if best_score == float('inf'):
                self.results["evaluation"].append(result) 
                continue

            self.compute_vi(evaluation, i, tweak_image_result)
            
            bias_cut = seg_params['bias_cut']
            for j in range(len(images)):
                if i == j:
                    continue

                image_name = self.config.image_names[j]
                other_image_result = result["evaluation_scores"][image_name]

                if bias_cut in used_bias_cuts:
                    evaluation = self.results["evaluation"][used_bias_cuts[bias_cut]]['evaluation_scores'][image_name]
                    evaluation = Evaluation.create(evaluation['under_segmentation'], evaluation['over_segmentation'], cached_seg)
                else:
                    evaluation = self.eval_image(images[j][1:], foreground_masks[j], bias_cut, j)
                evaluation.write_to_dict(other_image_result)
                self.compute_vi(evaluation, j, other_image_result)

                computed_VIs = [x['evaluation_scores'][image_name]['weighted_VI']
                                for x in self.results['evaluation']]
                if image_name not in segmentations or evaluation.weighted_VI <= min(computed_VIs):
                    segmentations[image_name] = evaluation.segmentation

            used_bias_cuts[bias_cut] = i
            self.results["evaluation"].append(result) 

        prog.close()
        return segmentations

    def eval_image(self, affinities, foreground_mask, bias_cut, i):
        # compute segmentation
        seg = run_mws(affinities, self.config.offsets,
                      stride=(1, 1, 1),
                      foreground_mask=foreground_mask,
                      seperating_channel=self.config.seperating_channel,
                      invert_dam_channels=True,
                      randomize_bounds=False,
                      bias_cut=bias_cut)
        seg = skimage.measure.label(seg)
        if seg.max() <= 255:
            seg = seg.astype(np.uint8)
        elif seg.max() <= 65_535:
            seg = seg.astype(np.uint16)
        elif seg.max() <= 4_294_967_295:
            seg = seg.astype(np.uint32)

        # compute VI
        vi = WeightedSingletonVI(self.ground_truths[i], seg, self.vi_weights[i])
        return Evaluation(vi, seg)

    def compute_vi(self, evaluation, i, result_dict):
        vi = VariationOfInformation(self.ground_truths[i], evaluation.segmentation)
        result_dict['variation_of_information'] = vi.value()
        result_dict['under_segmentation'] = vi.valueFalseJoin()
        result_dict['over_segmentation'] = vi.valueFalseCut()

    def compute_foreground_metrics(self, image, i, result_dict):
        y_pred = (image[0] > self.config.bg_threshold).ravel()
        y_true = (self.ground_truths[i] != 0).ravel()

        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, pos_label=1, average='binary')
        acc = accuracy_score(y_true, y_pred)
        diff = ((image[0] - self.ground_truth_affinites[i][0]) ** 2).mean()

        result_dict['foreground_prec'] = prec
        result_dict['foreground_rec'] = rec
        result_dict['foreground_f1'] = f1
        result_dict['foreground_acc'] = acc
        result_dict['foreground_diff'] = diff

    def compute_affinity_metrics(self, image, i, result_dict):
        pred_affinities = image[1:][self.ground_truth_foregrounds[i]]
        true_affinities = self.ground_truth_affinites[i][1:][self.ground_truth_foregrounds[i]]

        diff = ((pred_affinities - true_affinities) ** 2).mean()
        result_dict['affinity_diff'] = diff

        for i in range(0, pred_affinities.shape[0]):
            diff = ((pred_affinities[i] - true_affinities[i]) ** 2).mean()
            result_dict[f'affinity_diff_{i}'] = diff

        pred_attractive_aff = pred_affinities >= 0.5
        true_attractive_aff = true_affinities >= 0.5
        prec, rec, f1, _ = precision_recall_fscore_support(true_attractive_aff, pred_attractive_aff, pos_label=1, average='binary')
        result_dict['affinity_prec'] = prec
        result_dict['affinity_rec'] = rec
        result_dict['affinity_f1'] = f1

    def eval_and_store(self, images):
        segmentations = self.find_segmentation_and_eval(images, True)

        json_string = json.dumps(self.results, indent=4, cls=NpEncoder)
        if self.config.save_results:
            with open(self.determine_file_path(), 'x') as f:
                f.write(json_string)
        else:
            print(json_string)

        if self.config.show_segmentation:
            viewer = neuroglancer.Viewer()
            for name, image in segmentations.items():
                show_image(viewer, image, name=name, segmentation=True)
            webbrowser.open(str(viewer), new=0, autoraise=True)
            input("Done?")

def find_segmentation_and_eval(images, config: BrainbowSegmentationConfig):
    evaluater = SEBrainbow(config)
    return evaluater.eval_and_store(images)

def one_step(config: BrainbowSegmentationConfig):
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
