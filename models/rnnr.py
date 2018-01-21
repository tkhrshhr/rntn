import logging

import chainer.functions as F

from models.baser import Baser

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)


class RNNr(Baser):
    def __init__(self, n_embed=12, n_rel=7, d=25, k=75, n_ope=2):
        Baser.__init__(self, n_embed, n_rel, d, k, n_ope)

    def _node(self, left, right, ope):
        """
        Recieve left and right vectors
        Return vectors of their destinations
        """
        # Calculate
        preact = self._affine(left, right, ope)
        composed = F.tanh(preact)

        return composed

    def _compare(self, s, o):
        """
        Receive: batch
        Return: self.k dimenstion's vectors
        """

        # Calculate
        preact = self._affinec(s, o)
        compared = F.tanh(preact)

        return compared
