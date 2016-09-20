import argparse
import logging
import os
import subprocess
import sys
import tempfile


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


# init()
sys.path.append('{}/pixel-NN-training'.format(SCRIPT_DIR))
logging.basicConfig(
    level=logging.INFO,
    format='[%(name)s] %(levelname)s %(message)s'
)
os.environ['PATH'] += '{}{}/pixel-NN-training'.format(
    os.pathsep,
    os.getcwd()
)

try:
    from evalNN_keras import eval_nn
    from trainNN_keras import train_nn
except ImportError:
    logging.warning('ImportError for evalNN_keras or trainNN_keras')


def input_number(data):

    data = data[0]

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
        '--input', '{}.number.training.root'.format(base),
        '--output', '{}.number.training.root_'.format(base),
    ])
    subprocess.check_call([
        'mv',
        '{}.number.training.root_'.format(base),
        '{}.number.training.root'.format(base),
    ])

    return '{}/{}'.format(os.getcwd(), base)


def input_pos(data, nparticle):

    data = data[0]

    nn_type = 'pos{}'.format(nparticle)

    logger = logging.getLogger('launch:input_pos{}'.format(nn_type))

    base = os.path.basename(data)

    logger.info('Creating input for number neural network')
    subprocess.check_call([
        'python2',
        'pixel-NN/scripts/Run.py',
        '--scandirs', data,
        '--submit-dir', 'submit_{}'.format(nn_type),
        '--driver', 'direct',
        '--overwrite',
        '--type', nn_type
    ])

    logger.info('resizing the dataset')
    subprocess.check_call([
        '{}/RootCoreBin/bin/x86_64-slc6-gcc49-opt/resizePixelDataset'.format(
            os.getcwd()
        ),
        '-n', '12000000',
        '{}.{}.training.root'.format(base, nn_type),
        'submit_{}/data-NNinput/{}.root'.format(nn_type, base)
    ])
    subprocess.check_call([
        '{}/RootCoreBin/bin/x86_64-slc6-gcc49-opt/resizePixelDataset'.format(
            os.getcwd()
        ),
        '-s', '12000000',
        '-n', '5000000',
        '{}.{}.test.root'.format(base, nn_type),
        'submit_{}/data-NNinput/{}.root'.format(nn_type, base)
    ])

    return '{}/{}'.format(os.getcwd(), base)


def input_pos1(data):
    return input_pos(data, 1)


def input_pos2(data):
    return input_pos(data, 2)


def input_pos3(data):
    return input_pos(data, 3)


def input_error(data):

    logger = logging.getLogger('launch:input_error')

    pos_data, pos_nn = data

    output = os.path.basename(pos_data).replace('.pos', '.error')
    nn_output = '{}.ttrained.root'.format(os.path.basename(pos_nn))

    logger.info('converting the position neural network to TTrainedNetwork format')
    subprocess.check_call([
        'python2',
        'pixel-NN-training/keras2ttrained.py',
        '--model', '{}.model.yaml'.format(pos_nn),
        '--weights', '{}.weights.hdf5'.format(pos_nn),
        '--normalization', '{}.normalization.txt'.format(pos_nn),
        '--output', nn_output
    ])

    if 'pos1' in pos_data:
        nbins = 30
    elif 'pos2' in pos_data:
        nbins = 25
    elif 'pos3' in pos_data:
        nbins = 20

    for dset in ['training', 'test']:
        logger.info('creating the %s set', dset)

        subprocess.check_call([
            'python2',
            'pixel-NN-training/errorNN_input.py',
            '--input', os.path.realpath('{}.{}.root'.format(pos_data, dset)),
            '--ttrained', nn_output,
            '--output', '{}.{}.root'.format(output, dset),
            '--nbins', str(nbins)
        ])


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
            structure=[25, 20],
            activation='sigmoid2',
            output_activation='sigmoid2',
            regularizer=1e-7,
            momentum=0.4,
            batch=60,
            min_epochs=50,
            max_epochs=100,
            verbose=True
        )

    return '{}/{}'.format(os.getcwd(), name)


def training_pos(name, data, nparticle):

    nn_type = 'pos{}'.format(nparticle)

    logger = logging.getLogger('launch:training_{}'.format(nn_type))

    if nn_type not in name:
        name += '_{}'.format(nn_type)

    with genconfig(nn_type) as cfg:
        logger.info('training %s neural network', nn_type)
        train_nn(
            training_input='{}.{}.training.root'.format(data, nn_type),
            validation_fraction=0.1,
            output=name,
            config=cfg.name,
            structure=[40, 20],
            activation='sigmoid2',
            output_activation='linear',
            regularizer=1e-7,
            momentum=0.3,
            batch=30,
            min_epochs=50,
            max_epochs=100,
            verbose=True
        )

    return '{}/{}'.format(os.getcwd(), name)


def training_pos1(name, data):
    return training_pos(name, data, 1)


def training_pos2(name, data):
    return training_pos(name, data, 2)


def training_pos3(name, data):
    return training_pos(name, data, 3)


def training_error(name, data, flavor):

    nn_type = 'error{}'.format(flavor)

    logger = logging.getLogger('launch:training_{}'.format(nn_type))

    if nn_type not in name:
        name += '_{}'.format(nn_type)

    with genconfig(nn_type) as cfg:
        logger.info('training %s neural network', nn_type)
        train_nn(
            training_input='{}.{}.training.root'.format(data, nn_type[:-1]),
            validation_fraction=0.1,
            output=name,
            config=cfg.name,
            structure=[15, 10],
            activation='sigmoid2',
            output_activation='sigmoid2',
            regularizer=1e-6,
            momentum=0.7,
            batch=50,
            min_epochs=50,
            max_epochs=100,
            verbose=True
        )

    return '{}/{}'.format(os.getcwd(), name)


def training_error1x(name, data):
    return training_error(name, data, '1x')


def training_error1y(name, data):
    return training_error(name, data, '1y')


def training_error2x(name, data):
    return training_error(name, data, '2x')


def training_error2y(name, data):
    return training_error(name, data, '2y')


def training_error3x(name, data):
    return training_error(name, data, '3x')


def training_error3y(name, data):
    return training_error(name, data, '3y')



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


def evaluation_pos(nn_data, test_data, name, nparticle):
    nn_type = 'pos{}'.format(nparticle)
    logger = logging.getLogger('launch:evaluation_{}'.format(nn_type))

    with genconfig(nn_type) as cfg:
        logger.info('evaluating performance of %s network', nn_type)
        output = '{}.db'.format(os.path.basename(nn_data))
        eval_nn(
            inputp='{}.{}.test.root'.format(test_data, nn_type),
            model='{}.model.yaml'.format(nn_data),
            weights='{}.weights.hdf5'.format(nn_data),
            config=cfg.name,
            output=output,
            normalization='{}.normalization.txt'.format(nn_data),
        )

    subprocess.check_call([
        'bash',
        'pixel-NN-training/test-driver',
        nn_type,
        output,
        output.replace('.db', '.root')
    ])

    logger.warning('figures not produced yet for %s position neural networks', name)


def get_args():
    args = argparse.ArgumentParser()
    args.add_argument('type', choices=['number', 'pos1', 'pos2', 'pos3', 'error', 'error1x', 'error1y', 'error2x', 'error2y', 'error3x', 'error3y'])
    args.add_argument('name')
    args.add_argument('data', nargs='+')
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

    if (args.do_training or args.do_evaluation) and args.type == 'error':
        logger.error('"error" type only used with do_inputs! use error{1x,2y,2x,2y,3x,3y} otherwise.')
        exit(1)

    if args.do_inputs:
        input_f = globals()['input_{}'.format(args.type)]
        data = input_f(args.data)
    if args.do_training:
        training_f = globals()['training_{}'.format(args.type)]
        nn_data = training_f(args.name, data)
    if args.do_evaluation:
        evaluation_f = globals()['evaluation_{}'.format(args.type)]
        evaluation_f(nn_data, data, args.name)

    return 0


if __name__ == '__main__':
    main()
