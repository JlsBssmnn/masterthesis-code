def threshold(image, threshold=127.5):
  image[image < threshold] = 0
  image[image >= threshold] = 255
  return image