import argparse
from training_parser import parse_losses
from visualizer import plot_losses
from image_viewer import image_viewer as image_viewer_function
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).parent.parent.resolve()))
from misc import StoreDictKeyPair

def parse_and_plot_losses(opt):
    losses = parse_losses(opt.file)
    plot_losses(losses, opt)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(required=True)
    loss_plot_parser = subparsers.add_parser('losses')
    loss_plot_parser.set_defaults(func=parse_and_plot_losses)

    loss_plot_parser.add_argument('file', help='The path to the file that shall be visualized')
    loss_plot_parser.add_argument('--iteration_variable', default='iters', help='The name of the iteration variable that is used for the x-axis')
    loss_plot_parser.add_argument('--omit', default=[], nargs='*', help='Names of losses that should not be displayed')
    loss_plot_parser.add_argument('--show_only', default=None, nargs='+', help='If specified only these losses will be shown')
    loss_plot_parser.add_argument('--setting', default=None, help='The settings file that shall be applied')
    loss_plot_parser.add_argument('--style', default=None, help='The style that shall be applied')
    loss_plot_parser.add_argument('--output_file', default=None, help='If specified save figure to that file')

    image_viewer = subparsers.add_parser('images')
    image_viewer.set_defaults(func=image_viewer_function)
    image_viewer.add_argument('file', help='The path to the h5 file that contains the images')
    image_viewer.add_argument('--image_types', default=None, nargs='+', help='Which types of images are shown (e.g. fake_A or real_B)')
    image_viewer.add_argument('--setting', default=None, type=str, help='One of the defined settings to change the settings of neuroglancer')
    image_viewer.add_argument('--setting_options', default=dict(), action=StoreDictKeyPair, nargs="+", metavar="KEY=VAL", help='Options that are passed to the setting')
    image_viewer.add_argument('--no_open', action='store_true', help='If true, the browser is not opended automatically')

    group_identifier = image_viewer.add_mutually_exclusive_group()
    group_identifier.add_argument('--iterations', type=int, default=None, nargs='+', help='Which iterations of the training process are visualized')
    group_identifier.add_argument('--group_indices', type=int, default=None, nargs='+', help='Indices of groups in the h5 file that shall be visualized')


    opt = parser.parse_args()
    opt.func(opt)
