import numpy as np
from bbopt.lamcts import Lamcts
import matplotlib.pyplot as plt


class Levy:
    def __init__(self, dim=10):
        self.dim = dim
        self.lb = -5 * np.ones(dim)
        self.ub = 10 * np.ones(dim)

    def __call__(self, x):
        assert len(x) == self.dim
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        w = 1 + (x - 1.0) / 4.0
        val = np.sin(np.pi * w[0]) ** 2 + \
            np.sum((w[1:self.dim - 1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[1:self.dim - 1] + 1) ** 2)) + \
            (w[self.dim - 1] - 1) ** 2 * \
            (1 + np.sin(2 * np.pi * w[self.dim - 1])**2)
        return val


if __name__ == '__main__':
    f = Levy(20)
    lamcts = Lamcts(f, f.lb, f.ub, theta=50, cp=10)
    lamcts.run(100)
