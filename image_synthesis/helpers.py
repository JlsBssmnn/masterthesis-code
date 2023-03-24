import sys

from brainbows.cluster_brainbow_colors import cluster_brainbow_colors
from brainbows.colorize_brainbows import colorize_brainbows_cmap

if __name__ == '__main__':
  if len(sys.argv) < 2:
    print('You must provide the name of the helper you wish to invoke')
  elif sys.argv[1] == 'cluster_brainbow_colors':
    del sys.argv[1]
    cluster_brainbow_colors()
  elif sys.argv[1] == 'colorize_brainbows_cmap':
    del sys.argv[1]
    assert len(sys.argv) >= 6
    colorize_brainbows_cmap(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
