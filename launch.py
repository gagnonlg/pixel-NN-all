import os
import sys
import tempfile

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


def input_pos(data, nparticle):

    nn = 'pos{}'.format(nparticle)

    logger = logging.getLogger('launch:input_pos{}'.format(nn))

    base = os.path.basename(data)

    logger.info('Creating input for number neural network')
    subprocess.check_call([
        'python2',
        'pixel-NN/scripts/Run.py',
        '--scandirs', data,
        '--submit-dir', 'submit_{}'.format(nn),
        '--driver', 'direct',
        '--overwrite',
        '--type', nn
    ])

    logger.info('resizing the dataset')
    subprocess.check_call([
        '{}/RootCoreBin/bin/x86_64-slc6-gcc49-opt/resizePixelDataset'.format(os.getcwd()),
        '-n', '12000000',
        '{}.{}.training.root'.format(base, nn),
        'submit_{}/data-NNinput/{}.root'.format(nn, base)
    ])
    subprocess.check_call([
        '{}/RootCoreBin/bin/x86_64-slc6-gcc49-opt/resizePixelDataset'.format(os.getcwd()),
        '-s', '12000000',
        '-n', '5000000',
        '{}.{}.test.root'.format(base, nn),
        'submit_{}/data-NNinput/{}.root'.format(nn, base)
    ])

    return '{}/{}'.format(os.getcwd(), base)

def input_pos1(data):
    return input_pos(data, 1)
def input_pos2(data):
    return input_pos(data, 2)
def input_pos3(data):
    return input_pos(data, 3)


def genconfig(nn_type):

    logger = logging.getLogger('launch:genconfig')
    logger.info('generating variables configuration')

    tmp = tempfile.NamedTemporaryFile()
    subprocess.check_call(
        ['python2', 'pixel-NN-training/genconfig.py', '--type', nn_type],
        stdout=tmp
    )
    tmp.flush()
    return tmp

def training_number(name, data):
    logger = logging.getLogger('launch:training_number')

    if 'number' not in name:
        name += '_number'

    with genconfig('number') as cfg:
        logger.info('training number neural network')
        train_nn(
            training_input='{}.number.training.root'.format(data),
            validation_fraction=0.1,
            output=name,
            config=cfg.name,
            structure=[25,20],
            activation='sigmoid2',
            output_activation='sigmoid2',
            regularizer=1e-7,
            momentum=0.4,
            min_epochs=1,
            max_epochs=1,
            verbose=True
        )

    return '{}/{}'.format(os.getcwd(), name)

def evaluation_number(nn_data, test_data, name):
    logger = logging.getLogger('launch:evaluation_number')
    with genconfig('number') as cfg:
        logger.info('evaluating performance of number network')
        output = '{}.db'.format(os.path.basename(nn_data))
        eval_nn(
            inputp='{}.number.test.root'.format(test_data),
            model='{}.model.yaml'.format(nn_data),
            weights='{}.weights.hdf5'.format(nn_data),
            config=cfg.name,
            output=output,
            normalization='{}.normalization.txt'.format(nn_data),
        )

    os.environ['PATH'] += '{}{}/pixel-NN-training'.format(os.pathsep, os.getcwd())
    subprocess.check_call([
        'bash',
        'pixel-NN-training/test-driver',
        'number',
        output,
        output.replace('.db', '.root')
    ])

    subprocess.check_call([
        'python2',
        'pixel-NN-training/graphs/ROC_curves.py',
        output.replace('.db', '.root'),
        name
    ])

def get_args():
    args = argparse.ArgumentParser()
    args.add_argument('type', choices=['number', 'pos1', 'pos2', 'pos3'])
    args.add_argument('name')
    args.add_argument('data')
    args.add_argument('--do-inputs', default=False, action='store_true')
    args.add_argument('--do-training', default=False, action='store_true')
    args.add_argument('--do-evaluation', default=False, action='store_true')
    return args.parse_args()

def main():

    logger = logging.getLogger('launch:main')

    args = get_args()

    if not any([args.do_inputs]):
        logger.error('no action specified!')
        return 1

    if args.do_inputs:
        input_f = globals()['input_{}'.format(args.type)]
        data = input_f(args.data)
    if args.do_training:
        training_f = globals()['training_{}'.format(args.type)]
        nn_data = training_f(args.name, data)
    if args.do_evaluation:
        evaluation_f = globals()['evaluation_{}'.format(args.type)]
        eval_data = evaluation_f(nn_data, data, args.name)

    return 0


if __name__ == '__main__':
    main()
