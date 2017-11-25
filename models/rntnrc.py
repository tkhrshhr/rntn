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

            # Converter matrix (d -> n_rel)
            del self.C
            self.C = L.Linear(in_size=k, out_size=n_rel, initialW=u_initializer)

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
        s_vecs_re, s_vecs_im = F.split_axis(left, 2, axis=1)
        o_vecs_re, o_vecs_im = F.split_axis(right, 2, axis=1)

        # Concats of subject and object
        so_vecs_re = F.concat((s_vecs_re, s_vecs_im, o_vecs_re, o_vecs_im), axis=1)

        # Wr
        tensor_re = self.w_re[ope]
        tensor_im = self.w_im[ope]

        # Vr, br
        Vr = self.V[ope]
        br = self.b[ope]

        # Calculate each term
        # - sWo
        s_riri = F.concat([s_vecs_re, s_vecs_im, s_vecs_re, s_vecs_im], axis=1)
        o_riir = F.concat([o_vecs_re, o_vecs_im, o_vecs_im, o_vecs_re], axis=1)

        rr_ii_ri_ir = s_riri * o_riir  # element-wise product of s and o
        rr_ii_ri_ir_t = F.tile(rr_ii_ri_ir, (1, 2 * self.d))

        tensor_rrii_ = F.concat([tensor_re, tensor_re, tensor_im, tensor_im], axis=2)
        tensor_rrii = F.reshape(tensor_rrii_, (batch_size, 4 * self.d * 2 * self.d))

        sWo_re_ = F.sum(F.reshape(rr_ii_ri_ir_t * tensor_rrii, (batch_size, 2 * self.d, 4, self.d)), axis=3)
        rrr = sWo_re_[:, :, 0]
        rii = sWo_re_[:, :, 1]
        iri = sWo_re_[:, :, 2]
        iir = sWo_re_[:, :, 3]
        sWo_re = rrr + rii + iri - iir

        # Sum up terms
        rntn = sWo_re
        rnn = F.reshape(F.batch_matmul(Vr, so_vecs_re), (len(left), 2 * self.d)) + br

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
        s_vecs_re, s_vecs_im = F.split_axis(s, 2, axis=1)
        o_vecs_re, o_vecs_im = F.split_axis(o, 2, axis=1)

        # Repeat each subject and object vectors slice size times
        # - Concats of subject and object
        so_vecs_re = F.concat((s_vecs_re, s_vecs_im, o_vecs_re, o_vecs_im), axis=1)

        # Calculate each term
        # - sWo
        s_riri = F.concat([s_vecs_re, s_vecs_im, s_vecs_re, s_vecs_im], axis=1)
        o_riir = F.concat([o_vecs_re, o_vecs_im, o_vecs_im, o_vecs_re], axis=1)

        rr_ii_ri_ir = s_riri * o_riir  # element-wise product of s and o
        rr_ii_ri_ir_t = F.tile(rr_ii_ri_ir, (1, self.k))

        tensor_rrii_ = F.reshape(F.concat([self.wc_re, self.wc_re, self.wc_im, self.wc_im], axis=1), (1, 4 * self.d * self.k))
        tensor_rrii = F.tile(tensor_rrii_, (batch_size, 1))
        sWo_re_ = F.sum(F.reshape(rr_ii_ri_ir_t * tensor_rrii, (batch_size, self.k, 4, self.d)), axis=3)
        rrr = sWo_re_[:, :, 0]
        rii = sWo_re_[:, :, 1]
        iri = sWo_re_[:, :, 2]
        iir = sWo_re_[:, :, 3]
        sWo_re = rrr + rii + iri - iir

        rntn = sWo_re
        rnn = self.Vc(so_vecs_re)

        compared = F.tanh(rntn + rnn)

        return compared
