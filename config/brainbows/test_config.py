import numpy as np

class Config:
  branch_prob = 0.15
  image_size = (128, 256, 256)
  max_len = 150
  max_rotation = np.pi / 8
  max_thickness = 20
  min_len = 50
  min_thickness = 2
  neuron_count = 10
  noise_resolution_factor = 32
  perlin_max = 0.7
  retries = 10
  seed = 0

config = Config()
