import argparse
import importlib

from evaluation_utils import extend_path, verify_config

def config_verified(_):
    print('The configuration is correct ✅')

def perform_all_steps(config):
    images = translate_image.translate_image(config.translate_image_config)
    images, segmentation_config = create_segmentation.create_segmentation(images, config.segmentation_config)
    compute_eval.compute_eval(config, images, segmentation_config)

if __name__ == '__main__':
    extend_path()
    from evaluation import translate_image
    from evaluation import create_segmentation
    from evaluation import compute_eval

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)

    verify_parser = subparsers.add_parser('verify', help='Verifies a configuration. Found errors will be printed.')
    verify_parser.set_defaults(func=config_verified)
    verify_parser.add_argument('config', type=str, help='The config that shall be verified.')

    translate_image_parser = subparsers.add_parser('translate_image', aliases=['ti'], help='Applies a generator to a '
                                                   'given image and saves the output to a file')
    translate_image_parser.set_defaults(func=(lambda config: translate_image.one_step(config.translate_image_config)))
    translate_image_parser.add_argument('config', type=str, help='The config which shall be used')

    segmentation_tuner_parser = subparsers.add_parser('segmentation_tuner', help='Given network output, you can tweak the '
                                                      'settings for building a segmetation for this output.')
    segmentation_tuner_parser.set_defaults(func=(lambda config: create_segmentation.one_step(config.segmentation_config)))
    segmentation_tuner_parser.add_argument('config', type=str, help='The config which shall be used')

    compute_eval_parser = subparsers.add_parser('compute_eval', help='Given a segmentation and ground truth, '
                                                'evaluation metrics are computed')
    compute_eval_parser.set_defaults(func=compute_eval.one_step)
    compute_eval_parser.add_argument('config', type=str, help='The config which shall be used')

    evaluation_parser = subparsers.add_parser('evaluate', help='Evaluates a model. This performs all 3 stages.')
    evaluation_parser.set_defaults(func=perform_all_steps)
    evaluation_parser.add_argument('config', type=str, help='The config which shall be used')


    args = parser.parse_args()
    config = importlib.import_module(f'config.{args.config}').config
    verify_config(config)
    args.func(config)
