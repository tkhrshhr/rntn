import logging

import chainer
import chainer.functions as F
import chainer.links as L

from models.baser import Baser

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)


class RNTNrc(Baser):
    def __init__(self, n_embed=12, n_rel=7, d=25, k=75, n_ope=2):
        Baser.__init__(self, n_embed, n_rel, d, k)
        with self.init_scope():
            # Set initializer
            u_initializer = chainer.initializers.Uniform(dtype=self.xp.float32)
            initial_embed = self.xp.random.uniform(-0.01, 0.01, (n_embed, 2 * d))

            # Entity vectors
            del self.embed
            self.embed = L.EmbedID(n_embed, 2 * d, initialW=initial_embed)

            # RNTN layer
            # - Tensors W
            self.w_re = chainer.Parameter(shape=(n_ope, 2 * d, d), initializer=u_initializer)
            self.w_im = chainer.Parameter(shape=(n_ope, 2 * d, d), initializer=u_initializer)

            # - Standard layer V
            del self.V
            self.V = chainer.Parameter(shape=(n_ope, 2 * d, 4 * d), initializer=u_initializer)
            self.b = chainer.Parameter(shape=(n_ope, 2 * d), initializer=u_initializer)

            # Comparison layer
            # - Tensors W
            self.wc_re = chainer.Parameter(shape=(k, d), initializer=u_initializer)
            self.wc_im = chainer.Parameter(shape=(k, d), initializer=u_initializer)

            # - Standard layer V
            del self.Vc
            self.Vc = L.Linear(in_size=4 * d, out_size=k, initialW=u_initializer)

        # Other parameters
        self.comp = True

    def _node(self, left, right, ope):
        """
        Recieve left and right vectors
        Return vectors of their destinations
        """
        # Get batch size
        batch_size = len(left)

        # Get vectors of subjects and objects
        s_re, s_im = F.split_axis(left, 2, axis=1)
        o_re, o_im = F.split_axis(right, 2, axis=1)

        # Concats of subject and object
        so = F.concat((s_re, s_im, o_re, o_im), axis=1)

        # Wr
        w_re = self.w_re[ope]
        w_im = self.w_im[ope]

        # Vr, br
        Vr = self.V[ope]
        br = self.b[ope]

        # Calculate each term
        # - sWo
        s_riri = F.stack([s_re, s_im, s_re, s_im], axis=0)
        o_riir = F.stack([o_re, o_im, o_im, o_re], axis=0)

        s_dot_o = s_riri * o_riir  # element-wise product of s and o
        s_dot_o_t_ = F.tile(s_dot_o, (1, 1, 2 * self.d))
        s_dot_o_t = F.reshape(s_dot_o_t_, (4, batch_size, 2 * self.d, self.d))

        w_rrii = F.stack([w_re, w_re, w_im, w_im], axis=0)

        sWo_ = F.sum(s_dot_o_t * w_rrii, axis=3)
        sWo = sWo_[0] + sWo_[1] + sWo_[2] - sWo_[3]

        # Sum up terms
        rntn = sWo
        rnn = F.reshape(F.batch_matmul(Vr, so), (batch_size, 2 * self.d)) + br

        composed_re = F.tanh(rntn + rnn)

        return composed_re

    def _compare(self, s, o):
        """
        Receive: batch
        Return: self.n_rel-d vector
        """
        # Get batch size
        batch_size = len(s)

        # Get vectors of subjects and objects
        s_re, s_im = F.split_axis(s, 2, axis=1)
        o_re, o_im = F.split_axis(o, 2, axis=1)

        wc_re = F.tile(self.wc_re, (batch_size, 1, 1))
        wc_im = F.tile(self.wc_im, (batch_size, 1, 1))

        # Repeat each subject and object vectors slice size times
        # - Concats of subject and object
        so = F.concat((s_re, s_im, o_re, o_im), axis=1)

        # Calculate each term
        # - sWo
        s_riri = F.stack([s_re, s_im, s_re, s_im], axis=0)
        o_riir = F.stack([s_re, o_im, o_im, o_re], axis=0)

        s_dot_o = s_riri * o_riir  # element-wise product of s and o
        s_dot_o_t_ = F.tile(s_dot_o, (1, 1, self.k))
        s_dot_o_t = F.reshape(s_dot_o_t_, (4, batch_size, self.k, self.d))

        w_rrii = F.stack([wc_re, wc_re, wc_im, wc_im], axis=0)

        sWo_ = F.sum(s_dot_o_t * w_rrii, axis=3)
        sWo = sWo_[0] + sWo_[1] + sWo_[2] - sWo_[3]

        rntn = sWo
        rnn = self.Vc(so)

        compared = F.tanh(rntn + rnn)

        return compared
