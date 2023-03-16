import sys

from brainbows.cluster_brainbow_colors import cluster_brainbow_colors

if __name__ == '__main__':
  if len(sys.argv) < 2:
    print('You must provide the name of the helper you wish to invoke')
  elif sys.argv[1] == 'cluster_brainbow_colors':
    del sys.argv[1]
    cluster_brainbow_colors()