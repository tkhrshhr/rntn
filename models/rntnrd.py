import logging

import chainer
import chainer.functions as F

from models.baser import Baser

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)


class RNTNrd(Baser):
    def __init__(self, n_embed=12, n_rel=7, d=25, k=75, n_ope=2, mp=1):
        Baser.__init__(self, n_embed, n_rel, d, k)
        with self.init_scope():
            # Set initializer
            u_initializer = chainer.initializers.Uniform(scale=0.05, dtype=self.xp.float32)

            # RNTN layer
            # - Tensors W
            self.w = chainer.Parameter(shape=(n_ope, d, 1, d), initializer=u_initializer)

            # Comparison layer
            # - Tensors W
            self.wc = chainer.Parameter(shape=(k, 1, d), initializer=u_initializer)

        self.mp = mp

    def _node(self, left, right, ope):
        """
        Recieve left and right vectors
        Return vectors of their destinations
        """
        # Get batch size
        batch_size = len(left)

        # W
        tensor_t = self.w[ope]

        # Calculate each term
        # -sWo
        so_elp = left * right  # element-wise product of s and o
        so_elp_t = F.reshape(F.tile(so_elp, (1, self.d)), (batch_size, self.d, 1, self.d))

        sWo = F.sum(tensor_t * so_elp_t, (2, 3))
        affine = self._affine(left, right, ope, mp=self.mp)
        preact = sWo + affine

        # Activation
        composed = F.tanh(preact)

        return composed

    def _compare(self, s, o):
        """
        Receive: batch
        Return: self.n_rel-d vector
        """
        # Store batch size
        batch_size = len(s)

        # Repeat each subject and object vectors slice size times

        # W
        tensor_t = F.tile(self.wc, (batch_size, 1, 1, 1))

        # Calculate each term
        # - sWo
        so_elp = s * o  # element-wise product of s and o
        so_elp_t = F.reshape(F.tile(so_elp, (1, self.k)), (batch_size, self.k, 1, self.d))

        sWo = F.sum(tensor_t * so_elp_t, (2, 3))
        affine = self._affinec(s, o, mp=self.mp)
        preact = sWo + affine

        compared = F.tanh(preact)

        return compared
