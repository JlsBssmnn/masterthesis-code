import re
import h5py
import numpy as np
import sys
import pathlib
import webbrowser
from natsort import natsorted, ns

sys.path.append(str(pathlib.Path(__file__).parent.parent.resolve()))
from neuroglancer_viewer.neuroglancer_viewer import neuroglancer_viewer

def image_viewer(opt):
    with h5py.File(opt.file) as f:
        all_groups = natsorted(list(f.keys()), alg=ns.IGNORECASE)

        if opt.iterations is not None:
            groups = [x for x in all_groups if int(x[x.find('iter_') + 5:]) in opt.iterations]
        elif opt.group_indices is not None:
            groups = []
            for i in opt.group_indices:
                groups.append(all_groups[i])
        else:
            groups = all_groups

        assert len(groups) != 0, "No groups were selected for the given file"
        images = []
        names = []
        scales = []

        image_types = opt.image_types if opt.image_types is not None else list(f[groups[0]].keys())

        for group in groups:
            for image_type in image_types:
                dataset = f[group][image_type]
                epoch = re.search(r'epoch_(\d*)_iter', group).groups()[0]

                images.append(np.asarray(dataset))
                names.append(epoch + '_' + image_type)
                scales.append(tuple(dataset.attrs['element_size_um']))

    viewer = neuroglancer_viewer(images, names, scales, opt)
    print(viewer)
    if not opt.no_open:
        webbrowser.open(str(viewer), new=0, autoraise=True)
    input("Done?")
