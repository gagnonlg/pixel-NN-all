import os
import sys

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append('{}/pixel-NN-training'.format(script_dir))


import argparse
import logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(name)s] %(levelname)s %(message)s'
)
import subprocess

from evalNN_keras import eval_nn
from trainNN_keras import train_nn

def input_number(data):

    logger = logging.getLogger('launch:input_number')

    base = os.path.basename(data)

    logger.info('Creating input for number neural network')
    subprocess.check_call([
        'python2',
        'pixel-NN/scripts/Run.py',
        '--scandirs', data,
        '--submit-dir', 'submit_number',
        '--driver', 'direct',
        '--overwrite',
        '--type', 'number'
    ])

    logger.info('resampling the dataset')
    subprocess.check_call([
        'python2',
        'pixel-NN/scripts/balance_number.py',
        '--input', 'submit_number/data-NNinput/{}.root'.format(base),
        '--output', '{}.number'.format(base),
    ])

    logger.info('shuffling the training set')
    subprocess.check_call([
        'python2',
        'pixel-NN/scripts/shuffle_tree.py',
        '--seed', '750',
        '--input',  '{}.number.training.root'.format(base),
        '--output',  '{}.number.training.root_'.format(base),
    ])
    subprocess.check_call([
        'mv',
        '{}.number.training.root_'.format(base),
        '{}.number.training.root'.format(base),
    ])

    return '{}/{}'.format(os.getcwd(), base)


def get_args():
    args = argparse.ArgumentParser()
    args.add_argument('type', choices=['number'])
    args.add_argument('name')
    args.add_argument('data')
    args.add_argument('--do-inputs', default=False, action='store_true')
    #args.add_argument('--do-training', default=False, action='store_true')
    return args.parse_args()

def main():

    logger = logging.getLogger('launch:main')

    args = get_args()
    data = args.data

    if not any([args.do_inputs]):
        logger.error('no action specified!')
        return 1

    if args.do_inputs:
        input_f = globals()['input_{}'.format(args.type)]
        datadir = input_f(data)

    return 0


if __name__ == '__main__':
    main()


""" reserve

def train_number(name, data):
    with genconfig('number') as cfg:
        train_nn(
            training_input='{}.number.training.root'.format(data),
            validation_fraction=0.1,
            output=name,
            config=cfg.name,
            structure=[25,20],
            activation='sigmoid2',
            output_activation='sigmoid2',
            regularizer=FIXME,
            momentum=FIXME,
            min_epochs=50,
            max_epochs=100,
            verbose=True
        )

    return '{}/{}'.format(ospath.getcwd(), name)

def eval_number(nn_data, test_data):
    with tempfile.NamedTemporaryFile as tmp:
        subprocess.check_call(
            ['python2', 'genconfig.py', '--type', 'number']
            stdout=tmp
        )
        tmp.flush()

        eval_nn(
            inputp='{}.number.test.root'.format(test_data),
            model='{}.model.yaml'.format(nn_data),
            weights='{}.weights.model.yaml'.format(nn_data),
            config=tmp.name,
            output='{}.db'.format(os.path.basename(nn_data)),
            normalization='{}.normalization.txt'.format(nn_data),
        )


    if args.do_training:
        training_f = globals('training_{}'.format(args.type))
        datadir = training_f(name, datadir)
    if args.do_evaluation:
        evaluation_f = globals('evaluation_{}'.format(args.type))
        datadir = evaluation_f(name, datadir)
    if args.do_figures:
        figures_f = globals('figures_{}'.format(args.type))
        figures_f(name, datadir)


"""
