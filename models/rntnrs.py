import logging

import chainer
import chainer.functions as F

from models.baser import Baser

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)


class RNTNrs(Baser):
    def __init__(self, n_embed=12, n_rel=7, d=25, k=75, n_ope=2, l=1, m=1):
        Baser.__init__(self, n_embed, n_rel, d, k)
        with self.init_scope():
            # Set initializer
            u_initializer = chainer.initializers.Uniform(scale=0.05, dtype=self.xp.float32)

            # RNTN layer
            # - Tensors W
            self.S = chainer.Parameter(shape=(n_ope, d, d, l), initializer=u_initializer)
            self.T = chainer.Parameter(shape=(n_ope, d, l, d), initializer=u_initializer)
            # Comparison layer
            # - Tensors W
            self.Sc = chainer.Parameter(shape=(k, d, m), initializer=u_initializer)
            self.Tc = chainer.Parameter(shape=(k, m, d), initializer=u_initializer)

    def _node(self, left, right, ope):
        """
        Recieve left and right vectors
        Return vectors of their destinations
        """
        # Get vectors of subjects and objects
        s_vecs = left
        o_vecs = right

        # W
        S_t = self.S(ope)
        T_t = self.T(ope)

        # Calculate each term
        # -sWo
        sS = F.batch_matmul(s_vecs, S_t)
        To = F.batch_matmul(T_t, o_vecs)

        # Activation
        rntn = F.batch_matmul(sS, To)
        rnn = self._affine(left, right, ope)

        composed = F.tanh(rntn + rnn)

        return composed

    def _compare(self, s, o):
        """
        Receive: batch
        Return: self.n_rel-d vector
        """
        # Get batch_length
        batch_size = len(s)

        # - Concats of subject and object
        so_vecs = F.concat((s, o), axis=1)

        # W
        S_t = F.tile(self.Sc, (batch_size, 1))
        T_t = F.tile(self.Tc, (batch_size, 1))

        # Calculate each term
        # -sWo
        sS = F.batch_matmul(s, S_t)
        To = F.batch_matmul(T_t, o)

        # Activation
        rntn = F.batch_matmul(sS, To)
        rnn = self.Vc(so_vecs)

        compared = F.tanh(rntn + rnn)

        return compared
