import argparse
from yaml import safe_load
from trainer import train
from infer import test_single, test_all

if __name__ == '__main__':

    parser = argparse.ArgumentParser("Umbrella script for training / inference")
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Print debug info'
    )

    parser.add_argument('--config', '-c',
                        type=str, default='./config/tr_config.yml',
                        help='File path to training configuration .yml, default: ./config/tr_config.yml')

    subparsers = parser.add_subparsers(dest='algorithm', help='Training or Inference?')

    training = subparsers.add_parser('training', help='Proceed to training')
    training.add_argument('--ckpt', default=False, action='store_true',
                          help='Load from latest checkpoint to continue training, default: False')
    training.add_argument('--trial', default=False, action='store_true',
                          help='Enable TRIAL mode, for experimental training with smaller dataset splits')
    training.add_argument('--augment', default=False, action='store_true',
                          help='Enable rotation-based augmentation of scans in pre-processing')
    training.add_argument('--save', default=True, action='store_false',
                          help='Disable intermittent saving of weights as well as final model after training')

    inference = subparsers.add_parser('inference', help='Proceed to inference/evaluation')
    inference.add_argument('--file', '-f',
                           type=str, default=None,
                           help='Path to scan file to perform inference on, defaults to None --> random file chosen')
    inference.add_argument('--model', '-m',
                           type=str, default=None,
                           help='Path to model to perform inference with, defaults to None --> latest model chosen')
    inference.add_argument('--all', default=False, action='store_true',
                           help='Enable inference over all validation samples, displaying overall mIoU as output')
    inference.add_argument('--vis', default=False, action='store_true',
                           help='Enable visualization of single scan inferences')
    inference.add_argument('--ckpt', default=False, action='store_true',
                           help='Load model from latest checkpoint weights')
    inference.add_argument('--testds', default=False, action='store_true',
                           help='Perform inference on file from test split')

    FLAGS = parser.parse_args()

    if FLAGS.algorithm == 'training':
        print('==================================================================================')
        print('Training Mode')
        print('==================================================================================')
        train(FLAGS)

    elif FLAGS.algorithm == 'inference':
        print('==================================================================================')
        print('Inference Mode')
        print('==================================================================================')
        if FLAGS.all:
            test_all(FLAGS)
        else:
            test_single(FLAGS)
