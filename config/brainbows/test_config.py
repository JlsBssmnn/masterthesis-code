import numpy as np

class Config:
  branch_prob = 0.05
  branching_neuron_prob = 0.15
  cmap = './data/brainbow_colors.h5'
  cmap_dataset = 'colors'
  image_size = (128, 256, 256)
  large_rotation_prob = 0.05
  max_branch_rotation = np.pi / 2
  max_branches = 3
  max_len = 5
  max_rotation_large = np.pi / 6
  max_rotation_small = np.pi / 20
  max_thickness = [5, 15, 20, 25]
  max_thickness_probabilities = [0.1, 0.8, 0.05, 0.05]
  min_branch_rotation = np.pi / 4
  min_len = 2
  min_thickness = [1, 2, 3, 4]
  min_thickness_probabilities = [0.2, 0.65, 0.125, 0.025]
  min_voxel_per_neuron = 1000
  neuron_count = 10
  noise_resolution_factor = 32
  post_scaling = None
  refinement_noise_intensity = 2
  refinement_noise_resolution_factor = 8
  retries = 50
  seed = 0
  shortest_path_search_area = [40, 40, 40]
  termination_probability = 0.1

config = Config()
