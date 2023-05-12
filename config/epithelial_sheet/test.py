plane_dist = 16
z = 64

class BlobConfig:
  angle_noise = 0.01
  blob_max = 24 
  blob_min = 5
  first_plane_offset = plane_dist
  max_center_offset = [20, 20]
  max_child_count = 6
  max_dist = 18
  max_noise = 2
  max_offset_from_plane = plane_dist
  mean = 12
  min_child_count = 4
  min_dist = 10
  noise_resolution = (1, 1, 1)
  noise_shape = (16, 32, 32)
  plane_distance = plane_dist*2
  std = 3
  z_scaling = 0.5

class Config:
  angle_noise = 0.01
  blob_config = BlobConfig()
  cell_color = 0
  first_plane_offset = z / 2
  gaussian_sigma = 2.5
  limits = [z, 128, 128]
  max_center_offset = [10, 10]
  max_child_count = 5
  max_dist = 25
  max_offset_from_plane = z // 2
  membrane_color = 255
  membrane_threshold = 2.25
  min_child_count = 3
  min_dist = 15
  min_dist_local_max = 5
  noise_distribution_max = 1.5
  noise_intensity = 6
  noise_intensity_post = 0.5
  noise_resolution_factor = 8
  noise_resolution_post = (2, 8, 8)
  plane_distance = z + 1
  seed = 0

config = Config()
