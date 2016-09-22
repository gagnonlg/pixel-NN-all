import argparse
import logging
import os
import re
import subprocess
import sys
import tempfile


NN_TYPES = ['number', 'pos1', 'pos2', 'pos3', 'error1x', 'error1y',
            'error2x', 'error2y', 'error3x', 'error3y', 'pos',
            'error1', 'error2', 'error3']

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

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
    # pylint: disable=wrong-import-position
    from evalNN_keras import eval_nn
    from trainNN_keras import train_nn
except ImportError:
    logging.warning('ImportError for evalNN_keras or trainNN_keras')


def _validate_actions(actions):
    if not any([
            actions['do_inputs'],
            actions['do_training'],
            actions['do_eval'],
            actions['do_figures']
    ]):
        raise ValueError("No action specified")

    if actions['do_inputs'] and actions['do_eval'] and \
       not actions['do_training']:
        raise ValueError("Hole in sequence: do_training missing")

    if actions['do_training'] and actions['do_figures'] and \
       not actions['do_eval']:
        raise ValueError("Hole in sequence: do_eval missing")

    if actions['do_inputs'] and actions['do_figures'] and \
       not actions['do_training'] and not actions['do_eval']:
        raise ValueError("Hole in sequence: do_training and do_eval missing")


def _validate_type(actions, nn_type):
    if nn_type not in NN_TYPES:
        raise ValueError('Invalid nn_type: {}'.format(nn_type))

    if (actions['do_inputs'] or actions['do_training'] or actions['do_eval']) \
       and nn_type == 'pos':
        raise ValueError('nn_type "pos" can only be used with do_figures')

    if (actions['do_training'] or actions['do_eval'] or actions['do_figures']) \
       and nn_type in ['error1', 'error2', 'error3']:
        raise ValueError(
            'nn_type "%s" can only be used with do_inputs' % nn_type
        )

    if re.match('^pos[123]$', nn_type) and actions['do_figures']:
        raise ValueError(
            'nn_type "{}" can not be used with do_figures'.format(nn_type)
        )

    if re.match('^error[123][xy]$', nn_type) and actions['do_inputs']:
        raise ValueError(
            'nn_type "{}" can not be used with do_inputs'.format(nn_type)
        )


def _validate_data(actions, nn_type, data):  # pylint: disable=too-many-branches
    if re.match('^error[123]$', nn_type):
        if 'training-set' not in data:
            raise ValueError('--training-set <training set> not specified!')
        if 'test-set' not in data:
            raise ValueError('--test-set <test set> not specified!')
        if 'NN' not in data:
            raise ValueError(
                '--NN <model.yaml> <weights.hdf5> <normalization.txt>'
                ' not specified!'
            )

    elif nn_type == 'pos':
        if 'histograms' not in data or len(data['histograms']) != 3:
            raise ValueError(
                '--pos-histograms <pos1> <pos2> <pos3> not or over- specified!'
            )

    elif actions['do_inputs']:
        if 'AOD' not in data:
            raise ValueError('--AOD <aod> not specified!')

    elif actions['do_training']:
        if 'training-set' not in data:
            raise ValueError('--training-set <training set> not specified!')

    elif actions['do_eval']:
        if 'test-set' not in data:
            raise ValueError('--test-set <test set> not specified!')
        if 'NN' not in data:
            raise ValueError(
                '--NN <model.yaml> <weights.hdf5> <normalization.txt>'
                ' not specified!'
            )
    elif actions['do_figures']:
        if 'histograms' not in data or len(data['histograms']) != 1:
            raise ValueError(
                '--histograms <histo.root> not or over- specified!'
            )


def _validate_name(name):
    if not name:
        raise ValueError('Missing name')


def validate_plan(actions, data, nn_type, name):
    _validate_actions(actions)
    _validate_type(actions, nn_type)
    _validate_data(actions, nn_type, data)
    _validate_name(name)


def execute_plan(actions, data, nn_type, name):

    def const_data(*args):  # pylint: disable=unused-argument
        return data

    if actions['do_inputs']:
        input_f = globals()['input_{}'.format(nn_type)]
    else:
        input_f = const_data

    if actions['do_training']:
        training_f = globals()['training_{}'.format(nn_type)]
    else:
        training_f = const_data

    if actions['do_eval']:
        eval_f = globals()['eval_{}'.format(nn_type)]
    else:
        eval_f = const_data

    if actions['do_figures']:
        figures_f = globals()['figures_{}'.format(nn_type)]
    else:
        figures_f = const_data

    data = input_f(data)
    data = training_f(data, name)
    data = eval_f(data)
    figures_f(data)


def input_number(data):

    input_aod = data['AOD']

    logger = logging.getLogger('launch:input_number')

    base = os.path.basename(input_aod)

    logger.info('Creating input for number neural network')
    subprocess.check_call([
        'python2',
        'pixel-NN/scripts/Run.py',
        '--scandirs', input_aod,
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

    dset_base = '{}/{}'.format(os.getcwd(), base)
    data['training-set'] = '{}.number.training.root'.format(dset_base)
    data['test-set'] = '{}.number.test.root'.format(dset_base)

    return data


def input_pos(data, nparticle):

    input_aod = data['AOD']

    nn_type = 'pos{}'.format(nparticle)

    logger = logging.getLogger('launch:input_pos{}'.format(nn_type))

    base = os.path.basename(input_aod)

    logger.info('Creating input for number neural network')
    subprocess.check_call([
        'python2',
        'pixel-NN/scripts/Run.py',
        '--scandirs', input_aod,
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

    dset_base = '{}/{}'.format(os.getcwd(), base)
    data['training-set'] = '{}.pos{}.root'.format(dset_base, nparticle)
    data['test-set'] = '{}.pos{}.root'.format(dset_base, nparticle)

    return data


def input_pos1(data):
    return input_pos(data, 1)


def input_pos2(data):
    return input_pos(data, 2)


def input_pos3(data):
    return input_pos(data, 3)


def input_error(data, nparticle):

    logger = logging.getLogger('launch:input_error{}'.format(nparticle))

    nn_model, nn_weights, nn_norm = data['NN']

    nn_output = os.path.basename(nn_model).replace(
        'model.yaml',
        'ttrained.root'
    )

    logger.info(
        'converting the position neural network to TTrainedNetwork format'
    )
    subprocess.check_call([
        'python2',
        'pixel-NN-training/keras2ttrained.py',
        '--model', nn_model,
        '--weights', nn_weights,
        '--normalization', nn_norm,
        '--output', nn_output
    ])

    if nparticle == 1:
        nbins = 30
    elif nparticle == 2:
        nbins = 25
    elif nparticle == 3:
        nbins = 20

    for dkey in ['training-set', 'test-set']:
        logger.info('creating the %s', dkey)

        dset = data[dkey]
        output = os.path.basename(dset).replace('.pos', '.error')

        subprocess.check_call([
            'python2',
            'pixel-NN-training/errorNN_input.py',
            '--input', dset,
            '--ttrained', nn_output,
            '--output', output,
            '--nbins', str(nbins)
        ])

        data[dkey] = output

    return data


def input_error1(data):
    return input_error(data, 1)


def input_error2(data):
    return input_error(data, 2)


def input_error3(data):
    return input_error(data, 3)


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


def training_number(data, name):
    logger = logging.getLogger('launch:training_number')

    if 'number' not in name:
        name += '_number'

    with genconfig('number') as cfg:
        logger.info('training number neural network')
        train_nn(
            training_input=data['training-set'],
            validation_fraction=0.1,
            output=name,
            config=cfg.name,
            structure=[25, 20],
            activation='sigmoid2',
            output_activation='sigmoid2',
            regularizer=1e-7,
            momentum=0.4,
            batch=60,
            min_epochs=1,
            max_epochs=1,
            verbose=True
        )

    data_base = '{}/{}'.format(os.getcwd(), name)
    data['NN'] = [
        '{}.{}'.format(data_base, nn) for nn in
        ['model.yaml', 'weights.hdf5', 'normalization.txt']
    ]

    return data


def training_pos(data, name, nparticle):

    nn_type = 'pos{}'.format(nparticle)

    logger = logging.getLogger('launch:training_{}'.format(nn_type))

    if nn_type not in name:
        name += '_{}'.format(nn_type)

    with genconfig(nn_type) as cfg:
        logger.info('training %s neural network', nn_type)
        train_nn(
            training_input=data['training-set'],
            validation_fraction=0.1,
            output=name,
            config=cfg.name,
            structure=[40, 20],
            activation='sigmoid2',
            output_activation='linear',
            regularizer=1e-7,
            momentum=0.3,
            batch=30,
            min_epochs=1,
            max_epochs=1,
            verbose=True
        )

    data_base = '{}/{}'.format(os.getcwd(), name)
    data['NN'] = [
        '{}.{}'.format(data_base, nn) for nn in
        ['model.yaml', 'weights.hdf5', 'normalization.txt']
    ]

    return data


def training_pos1(data, name):
    return training_pos(data, name, 1)


def training_pos2(data, name):
    return training_pos(data, name, 2)


def training_pos3(data, name):
    return training_pos(data, name, 3)


def training_error(data, name, flavor):

    nn_type = 'error{}'.format(flavor)

    logger = logging.getLogger('launch:training_{}'.format(nn_type))

    if nn_type not in name:
        name += '_{}'.format(nn_type)

    with genconfig(nn_type) as cfg:
        logger.info('training %s neural network', nn_type)
        train_nn(
            training_input=data['training-set'],
            validation_fraction=0.1,
            output=name,
            config=cfg.name,
            structure=[15, 10],
            activation='sigmoid2',
            output_activation='sigmoid2',
            regularizer=1e-6,
            momentum=0.7,
            batch=50,
            min_epochs=1,
            max_epochs=1,
            verbose=True
        )

    data_base = '{}/{}'.format(os.getcwd(), name)
    data['NN'] = [
        '{}.{}'.format(data_base, nn) for nn in
        ['model.yaml', 'weights.hdf5', 'normalization.txt']
    ]

    return data


def training_error1x(data, name):
    return training_error(data, name, '1x')


def training_error1y(data, name):
    return training_error(data, name, '1y')


def training_error2x(data, name):
    return training_error(data, name, '2x')


def training_error2y(data, name):
    return training_error(data, name, '2y')


def training_error3x(data, name):
    return training_error(data, name, '3x')


def training_error3y(data, name):
    return training_error(data, name, '3y')


def _eval_nn(data, nn_type):
    logger = logging.getLogger('launch:evaluation_{}'.format(nn_type))
    with genconfig(nn_type) as cfg:
        logger.info('evaluating performance of %s network', nn_type)
        model, weights, norm = data['NN']
        output = os.path.basename(model).replace('model.yaml', 'db')
        eval_nn(
            inputp=data['test-set'],
            model=model,
            weights=weights,
            config=cfg.name,
            output=output,
            normalization=norm,
        )

    subprocess.check_call([
        'bash',
        'pixel-NN-training/test-driver',
        nn_type,
        output,
        output.replace('.db', '.root')
    ])

    data['histograms'] = ['{}/{}'.format(
        os.getcwd(),
        output.replace('.db', '.root')
    )]

    return data


def eval_number(data):
    return _eval_nn(data, 'number')


def eval_pos1(data):
    return _eval_nn(data, 'pos1')


def eval_pos2(data):
    return _eval_nn(data, 'pos2')


def eval_pos3(data):
    return _eval_nn(data, 'pos3')


def eval_error1x(data):
    return _eval_nn(data, 'error1x')


def eval_error1y(data):
    return _eval_nn(data, 'error1y')


def eval_error2x(data):
    return _eval_nn(data, 'error2x')


def eval_error2y(data):
    return _eval_nn(data, 'error2y')


def eval_error3x(data):
    return _eval_nn(data, 'error3x')


def eval_error3y(data):
    return _eval_nn(data, 'error3y')


def figures_number(data):
    logger = logging.getLogger('launch:figures_number')
    logger.info('producing figures for number neural network')
    histograms = data['histograms'][0]
    subprocess.check_call([
        'python2',
        'pixel-NN-training/graphs/ROC_curves.py',
        histograms,
        os.path.basename(histograms).replace('.root', '')
    ])


def figures_pos(data):
    logging.warning('figures_pos not implemented')


def figures_error1x(data):
    logging.warning('figures_error1x not implemented')


def figures_error1y(data):
    logging.warning('figures_error1y not implemented')


def figures_error2x(data):
    logging.warning('figures_error2x not implemented')


def figures_error2y(data):
    logging.warning('figures_error2y not implemented')


def figures_error3x(data):
    logging.warning('figures_error3x not implemented')


def figures_error3y(data):
    logging.warning('figures_error3y not implemented')


def launch(actions, data, nn_type, name):

    logger = logging.getLogger('launch')

    try:
        validate_plan(actions, data, nn_type, name)

    except ValueError as error:
        logger.error(error)
        exit(1)

    execute_plan(actions, data, nn_type, name)


def get_args():
    args = argparse.ArgumentParser()
    args.add_argument('nn_type')
    args.add_argument('nn_name')
    args.add_argument('--do-inputs', default=False, action='store_true')
    args.add_argument('--do-training', default=False, action='store_true')
    args.add_argument('--do-eval', default=False, action='store_true')
    args.add_argument('--do-figures', default=False, action='store_true')
    args.add_argument('--histograms', nargs='+')
    args.add_argument('--AOD')
    args.add_argument('--training-set')
    args.add_argument('--test-set')
    args.add_argument(
        '--NN',
        nargs=3,
        help='<model> <weights> <normalization>'
    )
    return args.parse_args()


def main():

    args = get_args()

    data = {}

    if args.AOD is not None:
        data['AOD'] = args.AOD
    if args.training_set is not None:
        data['training-set'] = args.training_set
    if args.test_set is not None:
        data['test-set'] = args.test_set
    if args.NN is not None:
        data['NN'] = args.NN
    if args.histograms is not None:
        data['histograms'] = args.histograms

    actions = {
        'do_inputs': args.do_inputs,
        'do_training': args.do_training,
        'do_eval': args.do_eval,
        'do_figures': args.do_figures
    }

    launch(actions, data, args.nn_type, args.nn_name)

if __name__ == '__main__':
    main()
