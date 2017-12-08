import numpy as np
import logging
import argparse

import chainer
from chainer import links as L
import joblib
from joblib import Parallel, delayed
from chainer import serializers


import reader
import linearizer
import converter

from models.rnn import RNN
from models.rntn import RNTN
from models.rntnd import RNTNd
from models.rntnc import RNTNc
from models.rnnr import RNNr
from models.rntnr import RNTNr
from models.rntnrd import RNTNrd
from models.rntnrc import RNTNrc
from models.rntnrs import RNTNrs


def parse_hyparams(model_name):
    hp_dict = {}

    hps = model_name.split('-')
    print(hps)
    hp_dict['r'] = int(hps[0][1])
    hp_dict['m'] = hps[1]
    for hp in hps[2:]:
        key = hp[0]
        value = hp[1:]
        if key == 'w':
            hp_dict[key] = float(value)
        else:
            print(key, value)
            hp_dict[key] = int(value)

    return hp_dict


def process(model, batch, batch_size):
    model(*c(batch))
    return model.accuracy.data * batch_size


def get_accuracy(model, data):
    # Prepare data
    data_s, data_o, data_r = data
    data_s_ld = [l(t, np) for t in data_s]
    data_o_ld = [l(t, np) for t in data_o]
    data_ld = np.array(chainer.datasets.TupleDataset(data_s_ld, data_o_ld, data_r))
    n_core = joblib.cpu_count()
    data_iter = np.array_split(data_ld, n_core)

    # Multiprocessing
    accs = Parallel(n_jobs=-1)([delayed(process)(model, b, len(b)) for b in data_iter])
    accuracy = np.sum(accs) / len(data_s)
    return accuracy


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--save', '-s', type=str, default='',
                        help='save name to test')
    args = parser.parse_args()

    # create logger with 'spam_application'
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # create file handler which logs even debug messages
    fh = logging.FileHandler('test_result/{}.log'.format(args.save), mode='w')
    fh.setLevel(logging.INFO)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    # Parse save file name
    hp_dict = parse_hyparams(args.save)
    train, dev, test = reader.read(hp_dict['x'], hp_dict['y'])

    # Prepare linearizer and converter
    global l, c

    # Model setup
    if hp_dict['r'] == 0:
        l = linearizer.linearize_tree
        c = converter.convert
        params = {'n_embed': hp_dict['x'] + 3, 'd': hp_dict['d'], 'k': hp_dict['k']}
        if hp_dict['m'] == 'n':
            model = RNN(**params)
        elif hp_dict['m'] == 't':
            model = RNTN(**params)
        elif hp_dict['m'] == 'd':
            model = RNTNd(**params)
        elif hp_dict['m'] == 'c':
            model = RNTNc(**params)

    elif hp_dict['r'] == 1:
        l = linearizer.linearize_tree_relational
        c = converter.convert_relational
        params = {'n_embed': hp_dict['x'] * 2, 'd': hp_dict['d'], 'k': hp_dict['k']}
        if hp_dict['m'] == 'rn':
            model = RNNr(**params)
        elif hp_dict['m'] == 'rt':
            model = RNTNr(**params)
        elif hp_dict['m'] == 'rd':
            model = RNTNrd(**params)
        elif hp_dict['m'] == 'rc':
            model = RNTNrc(**params)
        elif hp_dict['m'] == 'rs':
            model = RNTNrs(**params, p=hp_dict['p'], q=hp_dict['q'])

    print(args.save)
    serializers.load_hdf5("trained_model/" + args.save, model)
    model = L.Classifier(model)
    model.to_cpu()

    # Dev
    logger.info('---dev---')
    accuracy = get_accuracy(model, dev[:10])
    logger.info('dev: {}'.format(accuracy))

    # Test
    logger.info('---test---')
    for i, a_bin in enumerate(test[1:]):
        accuracy = get_accuracy(model, a_bin)
        logger.info('test {} : {}'.format(i + 1, accuracy))


if __name__ == '__main__':
    main()
