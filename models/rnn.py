import logging

import chainer.functions as F

from models.base import Base

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)


class RNN(Base):
    def __init__(self, n_embed=9, n_rel=7, d=25, k=75):
        Base.__init__(self, n_embed, n_rel, d, k)

    def _node(self, left, right):
        """
        Recieve left and right vectors
        Return vectors of their destinations
        """
        # Concats of subject and object
        so_vecs = F.concat((left, right), axis=1)

        # Calculate
        preact = self.V(so_vecs)
        composed = F.leaky_relu(preact, slope=0.01)

        return composed

    def _compare(self, s, o):
        """
        Receive: batch
        Return: self.k dimenstion's vectors
        """
        # Concats of subject and object
        so_vecs = F.concat((s, o), axis=1)

        # Calculate
        preact = self.Vc(so_vecs)
        compared = F.leaky_relu(preact)

        return compared
