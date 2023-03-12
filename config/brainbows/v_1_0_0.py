import numpy as np

class Config:
  branch_prob = 0.15
  image_size = (136, 512, 512)
  max_len = 150
  max_rotation = np.pi / 8
  max_thickness = 20
  min_len = 50
  min_thickness = 2
  neuron_count = 190
  noise_resolution_factor = 32
  perlin_max = 0.7
  retries = 50
  seed = 0

config = Config()
