import numpy as np
from .turbo1 import Turbo1


class Region:
    def __init__(self, node):
        self._node = node

    def is_within_region(self, x):
        child = self._node
        parent = self._node.parent
        while parent is not None:
            located_child = parent.which_child(x)
            if located_child is not child:
                return False
            child = parent
            parent = parent.parent
        return True


class TurboSampler:
    """Sampler for LA-MCTS(or any algorithm whose region is made by latent actions)
    """

    def __init__(self, f, node):
        self._f = f
        self._region = Region(node)
        self._node = node

    def sample(self):
        turbo1 = Turbo1(
            f=self._f,
            lb=self._f.lb,
            ub=self._f.ub,
            region = self._region,
            node = self._node,
            n_init=30,
            max_evals=40,
            batch_size=1
        )
        turbo1.optimize()
        return turbo1.X, turbo1.fX
