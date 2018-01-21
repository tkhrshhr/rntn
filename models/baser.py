import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L

import thin_stack as TS


class Baser(chainer.Chain):
    def __init__(self, n_embed=12, n_rel=7, d=25, k=75, n_ope=2):
        super().__init__()
        with self.init_scope():
            # Set initializer
            u_initializer_embed = chainer.initializers.Uniform(scale=0.01, dtype=self.xp.float32)
            u_initializer = chainer.initializers.Uniform(scale=0.05, dtype=self.xp.float32)

            # Entity vectors
            self.embed = L.EmbedID(in_size=n_embed, out_size=d, initialW=u_initializer_embed)

            # RNN layer
            self.V = chainer.Parameter(shape=(n_ope, d, 2 * d), initializer=u_initializer)
            self.b = chainer.Parameter(shape=(n_ope, d), initializer=u_initializer)

            # Comparison layer
            self.Vc = chainer.Parameter(shape=(k, 2 * d), initializer=u_initializer)
            self.bc = chainer.Parameter(shape=(k,), initializer=u_initializer)

            # Converter matrix (d -> n_rel)
            self.C = L.Linear(in_size=k, out_size=n_rel, initialW=u_initializer)

        # other hyparams
        self.n_embed = n_embed
        self.n_rel = n_rel
        self.d = d
        self.k = k
        self.opes = n_ope
        self.comp = False

    def _leaf(self, x):
        return self.embed(x)

    def _affine(self, left, right, ope, mp=1):
        # Concats of subject and object
        so_vecs = F.concat((left, right), axis=1)

        # Calculate each term
        Vr = self.V[ope]
        matpro = F.reshape(F.batch_matmul(Vr, so_vecs), (len(left), self.d))

        br = self.b[ope]

        # Calculate
        if mp == 1:
            affine = matpro + br
        elif mp == 0:
            affine = br

        return affine

    def _node(self, left, right, ope):
        raise NotImplementedError

    def _compose(self, batch):
        batch_size = int(len(batch) / 5)

        # -- Store Data
        lefts = batch[0: batch_size]
        rights = batch[batch_size: batch_size * 2]
        dests = batch[batch_size * 2: batch_size * 3]
        opes = batch[batch_size * 3: batch_size * 4]
        words = batch[batch_size * 4: batch_size * 5]

        # -- Sort all arrays in descending order and transpose them
        inds = np.argsort([-len(l) for l in lefts])
        root_inds = [len(words[i]) * 2 - 2 for i in inds]
        inds_reverse = [0] * batch_size
        for i, ind in enumerate(inds):
            inds_reverse[ind] = i

        lefts = F.transpose_sequence([lefts[i] for i in inds])
        rights = F.transpose_sequence([rights[i] for i in inds])
        dests = F.transpose_sequence([dests[i] for i in inds])
        opes = F.transpose_sequence([opes[i] for i in inds])
        words = F.transpose_sequence([words[i] for i in inds])

        # -- Store max length of sentence
        maxlen = len(words)

        # -- Calculate compositional vectors
        if self.comp:
            stack = self.xp.zeros((batch_size, maxlen * 2, self.d * 2), 'f')
        else:
            stack = self.xp.zeros((batch_size, maxlen * 2, self.d), 'f')

        for i, word in enumerate(words):
            batch = word.shape[0]
            es = self._leaf(word)
            ds = self.xp.full((batch,), i, 'i')
            stack = TS.thin_stack_set(stack, ds, es)

        for left, right, dest, ope in zip(lefts, rights, dests, opes):
            l, stack = TS.thin_stack_get(stack, left)
            r, stack = TS.thin_stack_get(stack, right)
            o = self._node(l, r, ope.data)
            stack = TS.thin_stack_set(stack, dest, o)

        lasts_ = stack[self.xp.arange(batch_size, dtype=self.xp.int32), root_inds]
        lasts = F.concat([F.expand_dims(lasts_[i], axis=0) for i in inds_reverse], axis=0)

        return lasts

    def _affinec(self, s, o, mp=1):
        # Concats of subject and object
        so = F.concat((s, o), axis=1)

        # Calculate each term
        matpro = F.matmul(so, self.Vc, transb=True)
        bias = F.broadcast_to(self.bc, shape=(len(s), self.k))

        if mp == 1:
            affine = matpro + bias
        elif mp == 0:
            affine = bias

        return matpro + affine

    def _compare(self, s, o):
        raise NotImplementedError

    def __call__(self, *inputs):
        """
        Receive batch
        Return output vectors
        """
        # Calculate each last compositional vector of S and O
        # - S
        s_composed = self._compose(inputs[0])
        # - O
        o_composed = self._compose(inputs[1])

        # Compare composed vectors
        compared = self._compare(s_composed, o_composed)

        # Last
        output = self.C(compared)
        return output
