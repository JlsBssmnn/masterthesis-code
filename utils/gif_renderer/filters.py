import numpy as np
from skimage.color import label2rgb

def threshold(image, threshold=127.5):
  '''Threshold the image.'''
  image[image < threshold] = 0
  image[image >= threshold] = 255
  return image

def invert(image):
    '''Inverts a uint8 image, so white -> black and black -> white. Similarly for gray values.'''
    return -image + 255

def color2label(image):
    '''Label a color image by assinging all voxels of the same color the same label. After creating
    the segmentation, the segmentation is visualized as a color image.'''
    c = image.shape[0]
    img = image.reshape(c, -1)

    new = np.zeros(image.shape[1:], dtype=int)
    img2 = img.reshape(c, -1)
    colors = np.unique(img2, axis=1)

    label = 1
    for i in range(colors.shape[1]):
        color = colors[:, i]
        mask = (img2.T == color).all(axis=1)
        mask = mask.reshape(*image.shape[1:])

        if (color == [0]*c).all():
            continue
        else:
            new[mask] = label
            label += 1

    image = label2rgb(new)
    image = (image * 255).astype(np.uint8)
    image = np.rollaxis(image, -1, 0)
    return image

def label(image):
    '''Visualizes a segmentation as a color image.'''
    image = label2rgb(image)
    image = (image * 255).astype(np.uint8)
    image = np.rollaxis(image, -1, 0)
    return image
