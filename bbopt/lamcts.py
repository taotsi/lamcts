"""
Learning Search Space Partition for Black-box Optimization using Monte Carlo Tree Search.
"""

import math
import sys

import numpy as np
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.quasirandom import SobolEngine

from .turbo.turbo_sampler import TurboSampler


class LatentAction:
    """
    Latent action representation.

    Attributes
    ----------
    svc: sklearn.svm.SVC
        The support vector machine classifier.
    good_label: int
        The label for high-performing region.
    bad_label: int
        The label for low-performing region.
    """
    def __init__(self):
        self.svc = None
        self.good_label = None
        self.bad_label = None

    def train(self, x_samples, y_samples):
        """Train to find SVM boundary."""
        kmeans = KMeans(n_clusters=2, tol=1e-7,
                        random_state=0).fit(np.c_[x_samples, y_samples])
        y = kmeans.labels_
        self.good_label = kmeans.cluster_centers_.argmin(0)[-1]
        self.bad_label = 1 - self.good_label

        # NOTE: we don't need to split the samples,
        #       since grid search will do the cross validation.
        # X_train, X_test, y_train, y_test = train_test_split(x_samples,
        #                                                     y,
        #                                                     test_size=0.3,
        #                                                     random_state=42)

        pipe = Pipeline([('scaler', StandardScaler()),
                         ('SVC', svm.SVC(class_weight='balanced'))])

        param_grid = {
            'SVC__C': np.logspace(-3, 4, 8),
            'SVC__gamma': np.logspace(-4, 3, 8)
        }
        grid = GridSearchCV(pipe,
                            param_grid,
                            refit=True,
                            n_jobs=-1,
                            scoring='balanced_accuracy')
        grid.fit(x_samples, y)

        self.svc = grid.best_estimator_

    def is_good(self, x):
        """Check a single sample is in high-performing region."""
        return self.svc.predict(x.reshape(1, -1))[0] == self.good_label


class Node:
    """
    Node representation in LA-MCTS tree.

    Attributes
    ----------
    parent: Node
        The parent node.
    left: Node
        The left child node.
    right: Node
        The right child node.
    visits: int
        Number of visits for UCB.
    value: number
        Accumulated value for UCB.
    latent_action:
        The SVM boundary that splits the region into two parts.
    """
    def __init__(self):
        self.parent = None
        self.left = None
        self.right = None
        self.visits = 0
        self.value = 0
        self.latent_action = LatentAction()

    def is_leaf(self):
        """Check whether this is a leaf node."""
        return self.left is None and self.right is None

    def select(self, cp):
        """Select a child with larger UCB score.

        Parameters
        ----------
        cp: float
            The UCB exploration parameter.
        """
        assert not self.is_leaf()
        ucbl = self.left.ucb(cp)
        ucbr = self.right.ucb(cp)
        return self.left if ucbl >= ucbr else self.right

    def ucb(self, cp):
        """Calculate node UCB score.

        Parameters
        ----------
        cp: float
            The exploration parameter.
        """
        epsilon = sys.float_info.epsilon
        exploitation = self.value / (self.visits + epsilon)
        exploration = 2 * cp * math.sqrt(
            2 * math.log(self.parent.visits + 1.0) / (self.visits + epsilon))
        return exploitation + exploration

    def add_samples(self, new_x_samples, new_y_samples):
        """Add new samples."""
        if hasattr(self, 'x_samples'):
            self.x_samples = np.vstack((self.x_samples, new_x_samples))
            self.y_samples = np.vstack((self.y_samples, new_y_samples))
        else:
            self.x_samples = new_x_samples
            self.y_samples = new_y_samples

    def good_samples(self):
        """Returns the good samples according to latent action."""
        indices = self.latent_action.svc.predict(
            self.x_samples) == self.latent_action.good_label
        return (self.x_samples[indices], self.y_samples[indices])

    def bad_samples(self):
        """Returns the bad samples according to latent action."""
        indices = self.latent_action.svc.predict(
            self.x_samples) != self.latent_action.good_label
        return (self.x_samples[indices], self.y_samples[indices])

    def which_child(self, x):
        if self.latent_action.is_good(x):
            return self.left
        else:
            return self.right


class Lamcts:
    """
    Latent Action Monte Carlo Tree Search.

    Optimize for minimum over a black box objective function,
    so transform your problem into a minimization problem first.

    Attributes
    ----------
    f: function
        The objective function.
    lb: numpy.array, shape (d,).
        Lower variable bounds.
    ub: numpy.array, shape (d,).
        Upper variable bounds.
    dim: int
        Dimension of the search space.
    f_best: number
        Best object value found.
    x_best: numpy.array, shape (n, d).
        Best solution found.
    root: Node
        The root of the search tree.
    theta: int
        The splitting threshold hyper-parameter.
    cp: float
        The UCB exploration hyper-parameter.
    """
    def __init__(self, f, lb, ub, theta, cp):
        self.f = f
        self.lb = lb
        self.ub = ub
        self.dim = len(lb)
        self.f_best = float('inf')
        self.root = Node()
        self.theta = theta
        self.cp = cp

    def split(self, node, new_x_samples, new_y_samples):
        """
        Dynamic tree construction via splitting.

        Splits the region represented by a node into a high-performing and
        a low-performing region, for its left and right child respectively.
        Split leaves recursively until no more leaves satisfy the splitting
        criterion.

        Parameters
        ----------
        node: Node
            A leaf node to split.
        x_samples: numpy.array
            New samples from the sampling stage.
        y_samples: numpy.array
            New evaluated value of black-box function on the samples.
        """
        assert node.is_leaf()

        if new_x_samples is not None:
            # implicit back propagation
            node.visits += len(new_x_samples)
            node.value += -new_y_samples.sum()
            node.add_samples(new_x_samples, new_y_samples)

        if node.visits >= self.theta:
            node.latent_action.train(node.x_samples, node.y_samples)

            left = Node()
            left.parent = node
            x_samples, y_samples = node.good_samples()
            # latent action should be trained properly
            assert len(x_samples) > 0 and len(x_samples) < len(node.x_samples)
            self.split(left, x_samples, y_samples)
            node.left = left

            right = Node()
            right.parent = node
            x_samples, y_samples = node.bad_samples()
            self.split(right, x_samples, y_samples)
            node.right = right

    def select(self):
        """Select via UCB."""
        node = self.root
        while not node.is_leaf():
            node = node.select(self.cp)
        return node

    def sample(self, node):
        """Sample via bayesian optimization."""
        assert node.is_leaf()
        # return self.random_sample(node)
        sampler = TurboSampler(self.f, node)
        return sampler.sample()

    def random_sample(self, node):
        """Naive random sample."""
        path = []
        n = node
        while n.parent:
            path.insert(0, (n.parent, n))
            n = n.parent

        if path:
            X = np.zeros((0, self.dim))
            is_left = np.array([(child == parent.left)
                                for parent, child in path])
            for x in node.x_samples:
                lb = np.clip(x - 1e-4, self.lb, self.ub)
                ub = np.clip(x + 1e-4, self.lb, self.ub)
                sobol = SobolEngine(dimension=self.dim, scramble=True)
                x_new = lb + (ub - lb) * sobol.draw(10).numpy()
                d = x_new - x

                for _ in range(16):
                    x_new += d
                    within_bound = ((self.lb <= x_new) &
                                    (x_new <= self.ub)).all(1)
                    is_good = np.array([
                        parent.latent_action.svc.predict(x_new) ==
                        parent.latent_action.good_label for parent, _ in path
                    ]).transpose()
                    mask = within_bound & (is_left == is_good).all(1)
                    cnt = np.count_nonzero(mask)
                    if cnt > 0:
                        x_new = x_new[mask]
                    if cnt <= 9:
                        break
                    d = d[mask] * 2

                X = np.vstack((X, x_new))
            X = X[np.random.choice(X.shape[0], size=10, replace=False)]

        else:
            X = np.random.uniform(self.lb, self.ub, (10, self.dim))

        Y = np.array([[self.f(x)] for x in X])
        best_index = Y.argmin()
        if Y[best_index][0] < self.f_best:
            self.f_best = Y[best_index][0]
            self.x_best = X[best_index]
            print("f_best: %s at \n%s" % (self.f_best, self.x_best))
        return X, Y

    def run(self, n_iterations):
        """Run n iterations."""
        leaf = self.root
        x_samples, y_samples = np.zeros((0, self.dim)), np.zeros((0, 1))
        for i in range(n_iterations):
            print("==================== LAMCTS iteration: {} =====================".format(i))
            self.split(leaf, x_samples, y_samples)
            leaf = self.select()
            x_samples, y_samples = self.sample(leaf)
