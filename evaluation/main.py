import argparse
import importlib
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).parent.parent))
sys.path.append(str(pathlib.Path(__file__).parent.parent / 'cycleGAN'))
from evaluation.config.template import Config, EpithelialSegmentationConfig
from evaluation import evaluate_brainbows, evaluate_epithelial, translate_image

from evaluation_utils import extend_path, verify_config

def config_verified(_):
    print('The configuration is correct ✅')

def segment_and_eval(config: Config):
    if isinstance(config.segmentation_config, EpithelialSegmentationConfig):
        evaluate_epithelial.one_step(config.segmentation_config)
    else:
        evaluate_brainbows.one_step(config.segmentation_config)

def perform_all_steps(config: Config):
    images = translate_image.translate_image(config.translate_image_config)

    if isinstance(config.segmentation_config, EpithelialSegmentationConfig):
        evaluate_epithelial.find_segmentation_and_eval(images, config.segmentation_config)
    else:
        evaluate_brainbows.find_segmentation_and_eval(images, config.segmentation_config)

if __name__ == '__main__':
    extend_path()

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)

    verify_parser = subparsers.add_parser('verify', help='Verifies a configuration. Found errors will be printed.')
    verify_parser.set_defaults(func=config_verified)
    verify_parser.add_argument('config', type=str, help='The config that shall be verified.')

    translate_image_parser = subparsers.add_parser('translate_image', aliases=['ti'], help='Applies a generator to a '
                                                   'given image and saves the output to a file')
    translate_image_parser.set_defaults(func=(lambda config: translate_image.one_step(config.translate_image_config)))
    translate_image_parser.add_argument('config', type=str, help='The config which shall be used')

    segmentation_tuner_parser = subparsers.add_parser('segment_and_eval', aliases=['se'], help='Given network output, find the best '
                                                      'segmetations and compute an evaluation for them.')
    segmentation_tuner_parser.set_defaults(func=segment_and_eval)
    segmentation_tuner_parser.add_argument('config', type=str, help='The config which shall be used')

    evaluation_parser = subparsers.add_parser('evaluate', help='Evaluates a model. This performs the 2 stages.')
    evaluation_parser.set_defaults(func=perform_all_steps)
    evaluation_parser.add_argument('config', type=str, help='The config which shall be used')


    args = parser.parse_args()
    config = importlib.import_module(f'config.{args.config}').config
    verify_config(config)
    args.func(config)
