import logging

import chainer
import chainer.functions as F

from models.baser import Baser

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)


class RNTNrs(Baser):
    def __init__(self, n_embed=12, n_rel=7, d=25, k=75, n_ope=2, p=1, q=1):
        Baser.__init__(self, n_embed, n_rel, d, k)
        with self.init_scope():
            # Set initializer
            u_initializer = chainer.initializers.Uniform(scale=0.05, dtype=self.xp.float32)

            # RNTN layer
            # - Tensors W
            self.S = chainer.Parameter(shape=(n_ope, d, d, p), initializer=u_initializer)
            self.T = chainer.Parameter(shape=(n_ope, d, p, d), initializer=u_initializer)
            # Comparison layer
            # - Tensors W
            self.Sc = chainer.Parameter(shape=(k, d, q), initializer=u_initializer)
            self.Tc = chainer.Parameter(shape=(k, q, d), initializer=u_initializer)

        self.p = p
        self.q = q

    def _node(self, left, right, ope):
        """
        Recieve left and right vectors
        Return vectors of their destinations
        """
        batch_size = len(left)

        left_t = F.reshape(F.tile(left, (1, self.d)), (batch_size * self.d, self.d))
        right_t = F.reshape(F.tile(right, (1, self.d)), (batch_size * self.d, self.d))

        # W
        S_t = F.reshape(self.S[ope], (batch_size * self.d, self.d, self.p))
        T_t = F.reshape(self.T[ope], (batch_size * self.d, self.p, self.d))

        # Calculate each term
        # -sWo
        sS = F.batch_matmul(left_t, S_t, transa=True)
        To = F.batch_matmul(T_t, right_t)

        # Activation
        rntn = F.reshape(F.batch_matmul(sS, To), (batch_size, self.d))
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

        s_t = F.reshape(F.tile(s, (1, self.k)), (batch_size * self.k, self.d))
        o_t = F.reshape(F.tile(o, (1, self.k)), (batch_size * self.k, self.d))

        # W
        S_t = F.reshape(F.tile(self.Sc, (batch_size, 1)), (batch_size * self.k, self.d, self.q))
        T_t = F.reshape(F.tile(self.Tc, (batch_size, 1)), (batch_size * self.k, self.q, self.d))

        # Calculate each term
        # -sWo
        sS = F.batch_matmul(s_t, S_t, transa=True)
        To = F.batch_matmul(T_t, o_t)

        # Activation
        rntn = F.reshape(F.batch_matmul(sS, To), (batch_size, self.k))
        rnn = self.Vc(so_vecs)

        compared = F.tanh(rntn + rnn)

        return compared
