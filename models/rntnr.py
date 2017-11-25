import logging

import chainer
import chainer.functions as F

from models.baser import Baser

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)


class RNTNr(Baser):
    def __init__(self, n_embed=12, n_rel=7, d=25, k=75, n_ope=2):
        Baser.__init__(self, n_embed, n_rel, d, k)
        with self.init_scope():
            # Set initializer
            u_initializer = chainer.initializers.Uniform(scale=0.05, dtype=self.xp.float32)

            # RNTN layer
            # - Tensors W
            self.W = chainer.Parameter(shape=(n_ope, d, d, d), initializer=u_initializer)

            # Comparison layer
            # - Tensors W
            self.Wc = chainer.Parameter(shape=(k, d, d), initializer=u_initializer)

    def _node(self, left, right, ope):
        """
        Recieve left and right vectors
        Return vectors of their destinations
        """
        # Get batch size
        batch_size = len(left)

        # Get vectors of subjects and objects
        s_vecs = left
        o_vecs = right

        # W
        tensor_t = self.W[ope]

        # Calculate each term
        # -sWo
        s_vecs_ = F.reshape(s_vecs, (batch_size, 1, self.d))
        s_vecs__ = F.broadcast_to(s_vecs_, ((batch_size, self.d, self.d)))
        o_vecs_ = F.reshape(o_vecs, (batch_size, self.d, 1))
        o_vecs__ = F.broadcast_to(o_vecs_, ((batch_size, self.d, self.d)))

        so_elp = s_vecs__ * o_vecs__  # element-wise product of s and o
        so_elp_t = F.reshape(F.tile(so_elp, (1, 1, self.d)), (batch_size, self.d, self.d, self.d))

        sWo = F.sum(tensor_t * so_elp_t, (2, 3))

        # Activation
        rntn = sWo
        rnn = self._affine(left, right, ope)

        composed = F.tanh(rntn + rnn)

        return composed

    def _compare(self, s, o):
        """
        Receive: batch
        Return: self.n_rel-d vector
        """
        # Store batch size
        batch_size = len(s)

        # Get vectors of subjects and objects
        s_vecs = s
        o_vecs = o

        # Repeat each subject and object vectors slice size times
        # - Concats of subject and object
        so_vecs = F.concat((s_vecs, o_vecs), axis=1)

        # W
        tensor_t = F.tile(self.Wc, (batch_size, 1, 1, 1))

        # Calculate each term
        # -sWo
        s_vecs_ = F.reshape(s_vecs, (batch_size, 1, self.d))
        s_vecs__ = F.broadcast_to(s_vecs_, ((batch_size, self.d, self.d)))
        o_vecs_ = F.reshape(o_vecs, (batch_size, self.d, 1))
        o_vecs__ = F.broadcast_to(o_vecs_, ((batch_size, self.d, self.d)))

        so_elp = s_vecs__ * o_vecs__  # element-wise product of s and o
        so_elp_t = F.reshape(F.tile(so_elp, (1, 1, self.k)), (batch_size, self.k, self.d, self.d))

        sWo = F.sum(tensor_t * so_elp_t, (2, 3))

        # Activation
        rntn = sWo
        rnn = self.Vc(so_vecs)

        compared = F.tanh(rntn + rnn)

        return compared
