import argparse
import importlib

from evaluation_utils import extend_path, verify_config

def config_verified(_):
    print('The configuration is correct ✅')

def perform_all_steps(config):
    images = translate_image.translate_image(config.translate_image_config)
    evaluate_image.find_segmentation_and_eval(images, config.segmentation_config)

if __name__ == '__main__':
    extend_path()
    from evaluation import translate_image
    from evaluation import evaluate_image

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
    segmentation_tuner_parser.set_defaults(func=(lambda config: evaluate_image.one_step(config.segmentation_config)))
    segmentation_tuner_parser.add_argument('config', type=str, help='The config which shall be used')

    evaluation_parser = subparsers.add_parser('evaluate', help='Evaluates a model. This performs the 2 stages.')
    evaluation_parser.set_defaults(func=perform_all_steps)
    evaluation_parser.add_argument('config', type=str, help='The config which shall be used')


    args = parser.parse_args()
    config = importlib.import_module(f'config.{args.config}').config
    verify_config(config)
    args.func(config)
