from sklearn import cluster
from skimage import io
import h5py
import numpy as np
import argparse
import os

from logging_config import logging

def cluster_brainbow_colors():
  parser = argparse.ArgumentParser()
  parser.add_argument('input_file', type=str, help='The input brainbow image')
  parser.add_argument('output_file', type=str, help='The file where to store the color values')
  parser.add_argument('--batch_size', type=int, default=None, help='How many voxels are being clustered at once')
  opt = parser.parse_args()

  f =  h5py.File(opt.output_file, 'w-')
  f.close()

  logging.info('Initialized options: %s, starting clustering', opt)

  algorithm = cluster.KMeans(n_clusters=190, n_init=10)
  brainbows = io.imread(opt.input_file)
  brainbows = brainbows.reshape(-1, 4)

  if opt.batch_size is not None:
    brainbows = brainbows[np.random.choice(brainbows.shape[0], opt.batch_size, replace=False)]

  res = algorithm.fit_predict(brainbows)
  logging.info('Finished clustering colors')

  colors = []
  for i in np.unique(res):
    colors.append(brainbows[res == i].mean(axis=0))
  colors = np.array(colors)
  logging.info('Finished averaging colors')

  with h5py.File(opt.output_file, 'a') as f:
    f.create_dataset('brainbow_colors', data=colors)
  logging.info('Color values written to %s. Script executed successfully',
    os.path.abspath(opt.output_file))