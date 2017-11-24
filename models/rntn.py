import logging

import chainer
import chainer.functions as F

from models.base import Base

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)


class RNTN(Base):
    def __init__(self, n_embed=9, n_rel=7, d=25, k=75):
        Base.__init__(self, n_embed, n_rel, d, k)
        with self.init_scope():
            # Set initializer
            u_initializer = chainer.initializers.Uniform(scale=0.05, dtype=self.xp.float32)

            # RNTN layer
            # - Tensors W
            self.W = chainer.Parameter(shape=(d, d, d), initializer=u_initializer)

            # Comparison layer
            # - Tensors W
            self.Wc = chainer.Parameter(shape=(k, d, d), initializer=u_initializer)

    def _node(self, left, right):
        """
        Recieve left and right vectors
        Return vectors of their destinations
        """
        # Get batch size
        batch_size = len(left)

        # Get vectors of subjects and objects
        s_vecs = left
        o_vecs = right

        # Repeat each subject and object vectors slice size times
        # - Concats of subject and object
        so_vecs = F.concat((s_vecs, o_vecs), axis=1)

        # W
        tensor_t = F.tile(self.W, (batch_size, 1, 1, 1))

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
        rnn = self.V(so_vecs)

        composed = F.leaky_relu(rntn, slope=0.01) + F.leaky_relu(rnn, slope=0.01)

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

        compared = F.leaky_relu(rntn, slope=0.01) + F.leaky_relu(rnn, slope=0.01)

        return compared
