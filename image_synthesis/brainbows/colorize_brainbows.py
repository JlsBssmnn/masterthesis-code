import numpy as np
import h5py

def extract_subbits(number, start, end, n_bits=None):
    assert start < end
    assert start >= 0 and end > 0

    if n_bits is None:
        n_bits = int(np.ceil(np.log2(number + 1)))
    number %= 2**(n_bits - start)
    number >>= n_bits - end
    return number

def colorize_brainbows_equally(input_file: str, output_file: str, dataset: str):
    """
    Colorizes brainbows by generating colors that are equally spaced when regarding colors as one number.
    (A color can be seen as one number if you concatenate it's bits together). The color values are
    generated via numpy's linspace function.

    Parameters
    -------
    input_file: The input brainbow image
    output_file: The file where to store the colorized brainbows
    dataset: The name of the dataset that contains the input image
    """
    with h5py.File(input_file, 'r') as f:
        brainbow_labels = np.asarray(f[dataset])
    labels = np.unique(brainbow_labels)

    n_colors = labels.shape[0]
    space = np.linspace(0, 2**32 - 1, n_colors).astype(np.uint32)

    brainbows = np.empty((4, *brainbow_labels.shape), dtype=np.uint8)
    for i, label in enumerate(sorted(labels)):
        mask = brainbow_labels == label
        brainbows[0, mask] = extract_subbits(space[i], 0, 8, 32)
        brainbows[1, mask] = extract_subbits(space[i], 8, 16, 32)
        brainbows[2, mask] = extract_subbits(space[i], 16, 24, 32)
        brainbows[3, mask] = extract_subbits(space[i], 24, 32, 32)

    with h5py.File(output_file, 'w-') as f:
        d = f.create_dataset('brainbows', data=brainbows)
        d.attrs['element_size_um'] = [0.2, 0.1, 0.1]


def colorize_brainbows_cmap(input_file: str, dataset: str, cmap_file: str, cmap_dataset: str):
    """
    Colorizes a labeled brainbow image with the given color map.

    Parameters
    -------
    input_file: The input brainbow image
    dataset: The name of the dataset that contains the input image
    cmap_file: The file that contains the colors that shall be used
    cmap_dataset: The name of the dataset in the cmap_file that contains the colors

    Returns
    -------
    The colorized image.
    """
    assert input_file.endswith('.h5') and cmap_file.endswith('.h5'), \
            "This function only operates on h5 files"
    with h5py.File(cmap_file, 'r') as f:
        colors = np.asarray(f[cmap_dataset])
    with h5py.File(input_file, 'r') as f:
        brainbow_labels = np.asarray(f[dataset])

    labels = sorted(np.unique(brainbow_labels))
    assert len(labels) <= colors.shape[0], "Not enough colors in cmap"

    if brainbow_labels.ndim == 3:
        image = np.empty((4, *brainbow_labels.shape), dtype=np.uint8)
    elif brainbow_labels.ndim == 4:
        new_shape = list(brainbow_labels.shape)
        new_shape[0] = 4
        image = np.empty(new_shape, dtype=np.uint8)
    else:
        raise Exception("Brainbow label image must be either 3D or 4D")

    for i, label in enumerate(labels): 
        mask = brainbow_labels == label
        for j in range(4):
            image[j, mask] = colors[i][j]

    return image
