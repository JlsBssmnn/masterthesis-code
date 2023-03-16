import numpy as np

class Config:
  branch_prob = 0.05
  branching_neuron_prob = 0.15
  image_size = (128, 256, 256)
  max_branches = 3
  max_len = 5
  max_rotation = np.pi / 6
  max_thickness = 20
  min_len = 2
  min_thickness = 2
  min_voxel_per_neuron = 1000
  neuron_count = 10
  noise_resolution_factor = 32
  perlin_max = 0.7
  retries = 10
  seed = 0

config = Config()
