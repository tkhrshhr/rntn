import numpy
import argparse
import datetime
import chainer
from chainer import cuda
from chainer import training
from chainer.training import extensions
from chainer import links as L

import matplotlib as mpl
mpl.use('Agg')

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--relational', '-r', type=int, default=0,
                        help='If model is relational')

    parser.add_argument('--n_var', '-x', type=int, default=6,
                        help='The number of propositional variable')

    parser.add_argument('--max_n_var', '-y', type=int, default=4,
                        help='The  max number of propositional variable in a formula')

    parser.add_argument('--model', '-m', type=str, default='n',
                        help='Model to train')

    parser.add_argument('--p_dim', '-p', type=int, default=1,
                        help='Dimension p')

    parser.add_argument('--q_dim', '-q', type=int, default=1,
                        help='Dimension q')

    parser.add_argument('--epoch', '-e', default=100, type=int,
                        help='number of epochs to learn')

    parser.add_argument('--dimension', '-d', type=int, default=35,
                        help='Dimension of embeddings')

    parser.add_argument('--output_vector', '-k', type=int, default=75,
                        help='Dimension of output vector')

    parser.add_argument('--weightdecay', '-w', type=float, default=0.0001,
                        help='Coefficient of weight decay')

    parser.add_argument('--batchsize', '-b', type=int, default=25,
                        help='learning minibatch size')

    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')

    parser.add_argument('--nmodifier', '-n', default='',
                        help='Result name modifier')

    parser.add_argument('--epocheval', '-v', type=int, default=2,
                        help='number of epochs per evaluation')

    parser.add_argument('--test', dest='test', action='store_true')

    parser.set_defaults(test=False)
    args = parser.parse_args()

    # Log file name setting
    today = datetime.date.today()
    month = today.month
    day = today.day
    resultname = "{}{}{}-b{}d{}w{}".format(month, day, args.nmodifier, args.batchsize, args.dimension, args.weightdecay)

    # Data setup
    train, dev, test = reader.read(args.n_var, args.max_n_var)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        xp = cuda.cupy
    else:
        xp = numpy

    # - Prepare train/test iterator
    if args.relational == 0:
        l = linearizer.linearize_tree
    elif args.relational == 1:
        l = linearizer.linearize_tree_relational
    # -- Train
    train_s, train_o, train_r = train
    train_s_ld = [l(t, xp) for t in train_s]
    train_o_ld = [l(t, xp) for t in train_o]
    train_datasets = chainer.datasets.TupleDataset(train_s_ld, train_o_ld, train_r)
    train_iter = chainer.iterators.SerialIterator(train_datasets, args.batchsize)

    # -- Test
    test_s, test_o, test_r = test
    test_s_ld = [l(t, xp) for t in test_s]
    test_o_ld = [l(t, xp) for t in test_o]
    test_datasets = chainer.datasets.TupleDataset(test_s_ld, test_o_ld, test_r)
    test_iter = chainer.iterators.SerialIterator(test_datasets, args.batchsize, repeat=False, shuffle=False)

    # Model setup
    if args.relational == 0:
        params = {'n_embed': args.n_var + 3, 'd': args.dimension, 'k': args.output_vector}
        if args.model == 'n':
            result_dir = 'result_rnn'
            model = RNN(**params)
        elif args.model == 't':
            result_dir = 'result_rntn'
            model = RNTN(**params)
        elif args.model == 'd':
            result_dir = 'result_rntnd'
            model = RNTNd(**params)
        elif args.model == 'c':
            result_dir = 'result_rntnc'
            model = RNTNc(**params)

    elif args.relational == 1:
        params = {'n_embed': args.n_var * 2, 'd': args.dimension, 'k': args.output_vector}
        if args.model == 'rn':
            result_dir = 'result_rnnr'
            model = RNNr(**params)
        elif args.model == 'rt':
            result_dir = 'result_rntnr'
            model = RNTNr(**params)
        elif args.model == 'rd':
            result_dir = 'result_rntnrd'
            model = RNTNrd(**params)
        elif args.model == 'rc':
            result_dir = 'result_rntnrc'
            model = RNTNrc(**params)
        elif args.model == 'rs':
            result_dir = 'result_rntnrs'
            resultname += 'p{}q{}'.format(args.p_dim, args.q_dim)
            model = RNTNrs(**params, p=args.p_dim, q=args.q_dim)

    if args.gpu >= 0:
        model.to_gpu()

    # Optimizer setup
    optimizer = chainer.optimizers.AdaDelta(eps=1e-07)
    optimizer.setup(L.Classifier(model))
    optimizer.add_hook(chainer.optimizer.WeightDecay(args.weightdecay))

    # Trainer setup
    if args.relational == 0:
        c = converter.convert
    elif args.relational == 1:
        c = converter.convert_relational

    # - Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu, converter=c)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=result_dir)

    # - Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, L.Classifier(model), device=args.gpu, converter=c))

    # - Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport(log_name="{}log".format(resultname)))

    # - Save two plot images to the result dir
    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                  'epoch', file_name='{}loss.png'.format(resultname)))
        trainer.extend(
            extensions.PlotReport(
                ['main/accuracy', 'validation/main/accuracy'],
                'epoch', file_name='{}accuracy.png'.format(resultname)))

    # - Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    # - Run trainer
    trainer.run()

    # - Save model
    chainer.serializers.save_hdf5("trained_model/{}_{}".format(args.model, resultname), model)


if __name__ == '__main__':
    main()
