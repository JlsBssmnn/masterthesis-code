from skimage.metrics import variation_of_information

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

def evaluate_segmentation_epithelial(segmentation, membrane_truth, cell_truth):
    segmentation_cells = (segmentation != 0)
    membrane_mask = segmentation_cells & (membrane_truth != 0)
    cell_mask = segmentation_cells & (cell_truth != 0)

    if membrane_mask.any():
        _, under_segmentation = variation_of_information(membrane_truth[membrane_mask], segmentation[membrane_mask], ignore_labels=[0])
    else:
        under_segmentation = float('nan')

    if cell_mask.any():
        over_segmentation, _ = variation_of_information(cell_truth[cell_mask], segmentation[cell_mask], ignore_labels=[0])
    else:
        over_segmentation = float('nan')

    cell_mask = cell_truth != 0
    pred_cells = segmentation[cell_mask] != 0
    acc = pred_cells.sum() / cell_mask.sum()

    return Evaluation(under_segmentation, over_segmentation, segmentation, acc)
