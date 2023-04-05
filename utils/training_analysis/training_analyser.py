import argparse
from training_parser import parse_losses
from visualizer import plot_losses

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='The path to the file that shall be visualized')
    parser.add_argument('--iteration_variable', default='iters', help='The name of the iteration variable that is used for the x-axis')
    parser.add_argument('--omit', default=[], nargs='*', help='Names of losses that should not be displayed')
    parser.add_argument('--show_only', default=None, nargs='+', help='If specified only these losses will be shown')
    parser.add_argument('--setting', default=None, help='The settings file that shall be applied')
    parser.add_argument('--style', default=None, help='The style that shall be applied')
    parser.add_argument('--output_file', default=None, help='If specified save figure to that file')

    opt = parser.parse_args()

    losses = parse_losses(opt.file)
    plot_losses(losses, opt)
