import chainer
import chainer.functions as F
import chainer.links as L


from models.base import Base


class RNTNc(Base):
    def __init__(self, n_embed=9, n_rel=7, d=25, k=75):
        """
        d: embedding size
        """
        Base.__init__(self, n_embed, n_rel, d, k)
        with self.init_scope():
            # Set initializer
            u_initializer = chainer.initializers.Uniform(dtype=self.xp.float32)
            initial_embed = self.xp.random.uniform(-0.01, 0.01, (n_embed, 2 * d))

            # Entity vectors
            del self.embed
            self.embed = L.EmbedID(n_embed, 2 * d, initialW=initial_embed)

            # RNTN layer
            # - Tensors W
            self.w_re = chainer.Parameter(shape=(2 * d, d), initializer=u_initializer)
            self.w_im = chainer.Parameter(shape=(2 * d, d), initializer=u_initializer)

            # - Standard layer V
            del self.V
            self.V = L.Linear(in_size=4 * d, out_size=2 * d, initialW=u_initializer)

            # Comparison layer
            # - Tensors W
            self.wc_re = chainer.Parameter(shape=(2 * k, d), initializer=u_initializer)
            self.wc_im = chainer.Parameter(shape=(2 * k, d), initializer=u_initializer)

            # - Standard layer V
            del self.Vc
            self.Vc = L.Linear(in_size=4 * d, out_size=2 * k, initialW=u_initializer)

            # Converter matrix (d -> n_rel)
            del self.C
            self.C = L.Linear(in_size=2 * k, out_size=n_rel, initialW=u_initializer)

        # Other parameters
        self.comp = True

    def _node(self, left, right):
        """
        Recieve left and right vectors
        Return vectors of their destinations
        """
        # Get batch size
        batch_size = len(left)

        # Get vectors of subjects and objects
        s_vecs_re, s_vecs_im = F.split_axis(left, 2, axis=1)
        o_vecs_re, o_vecs_im = F.split_axis(right, 2, axis=1)

        # Repeat each subject and object vectors slice size times
        # - Concats of subject and object
        so_vecs_re = F.concat((s_vecs_re, s_vecs_im, o_vecs_re, o_vecs_im), axis=1)

        # W
        tensor_re_t = F.tile(self.w_re, (batch_size, 1, 1))
        tensor_im_t = F.tile(self.w_im, (batch_size, 1, 1))

        # Calculate each term
        # - sWo
        s_riri = F.stack([s_vecs_re, s_vecs_im, s_vecs_re, s_vecs_im], axis=1)
        o_riir = F.stack([o_vecs_re, o_vecs_im, o_vecs_im, o_vecs_re], axis=1)

        rr_ii_ri_ir = s_riri * o_riir  # element-wise product of s and o
        rr_ii_ri_ir_t = F.reshape(F.tile(rr_ii_ri_ir, (1, 2 * self.d, 1)), (batch_size, 2 * self.d, 4, self.d))

        tensor_rrii = F.stack([tensor_re_t, tensor_re_t, tensor_im_t, tensor_im_t], axis=2)
        sWo_re = F.sum(rr_ii_ri_ir_t * tensor_rrii, axis=(2, 3))

        # Sum up terms
        rntn = sWo_re
        rnn = self.V(so_vecs_re)

        composed_re = F.leaky_relu(rntn, slope=0.01) + F.leaky_relu(rnn, slope=0.01)

        return composed_re

    def _compare(self, s, o):
        """
        Recieve left and right vectors
        Return vectors of their destinations
        """
        # Get batch size
        batch_size = len(s)

        # Get vectors of subjects and objects
        s_vecs_re, s_vecs_im = F.split_axis(s, 2, axis=1)
        o_vecs_re, o_vecs_im = F.split_axis(o, 2, axis=1)

        # Repeat each subject and object vectors slice size times
        # - Concats of subject and object
        so_vecs_re = F.concat((s_vecs_re, s_vecs_im, o_vecs_re, o_vecs_im), axis=1)

        # W
        tensor_re_t = F.tile(self.wc_re, (batch_size, 1, 1))
        tensor_im_t = F.tile(self.wc_im, (batch_size, 1, 1))

        # Calculate each term
        # - sWo
        s_riri = F.stack([s_vecs_re, s_vecs_im, s_vecs_re, s_vecs_im], axis=1)
        o_riir = F.stack([o_vecs_re, o_vecs_im, o_vecs_im, o_vecs_re], axis=1)

        rr_ii_ri_ir = s_riri * o_riir  # element-wise product of s and o
        rr_ii_ri_ir_t = F.reshape(F.tile(rr_ii_ri_ir, (1, 2 * self.k, 1)), (batch_size, 2 * self.k, 4, self.d))

        tensor_rrii = F.stack([tensor_re_t, tensor_re_t, tensor_im_t, tensor_im_t], axis=2)
        sWo_re = F.sum(rr_ii_ri_ir_t * tensor_rrii, axis=(2, 3))

        # Sum up terms
        rntn = sWo_re
        rnn = self.Vc(so_vecs_re)

        compared_re = F.leaky_relu(rntn, slope=0.01) + F.leaky_relu(rnn, slope=0.01)

        return compared_re
