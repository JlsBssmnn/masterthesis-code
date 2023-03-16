import numpy as np

class Config:
  branch_prob = 0.05
  branching_neuron_prob = 0.15
  image_size = (136, 512, 512)
  max_branches = 3
  max_len = 10
  max_rotation = np.pi / 6
  max_thickness = 20
  min_len = 4
  min_thickness = 2
  min_voxel_per_neuron = 1000
  neuron_count = 190
  noise_resolution_factor = 32
  perlin_max = 0.7
  retries = 500
  seed = 0

config = Config()
