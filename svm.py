"""
.. module:: svm
   :synopsis: This file contains a linear SVM solver.
.. moduleauthor:: Angelo Ortiz <github.com/angelo-ortiz>
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import layer_norm, extract_sq_patches, gaussian_window

class LinearSVM:
    def __init__(self, C=1., tol=1e-3, max_iter=-1):
        super().__init__()
        self.C = X
        self.tol = tol
        self.max_iter = max_iter

    def init_multipliers(self):
        self._alphas = None

    def _take_step(self, i1, i2):
        if i1 == i2:
            return False

        alph1 = self._alphas[i1]
        alph_gap = self._alphas[i2] - alph1
        y1 = self._y[i1]  # TODO
        e1 = self._errors[i1]
        l = max(0, alph_gap)
        h = min(self.C, ...) # TODO

        if l == h:
            return False

        pass

    def _examine_example(self, i):
        r_i = self._y[i] * self._errors.get(i)

        if (r_i < -self.tol and self._alphas[i] < self.C) \
           or (r_i > self.tol and self._alphas[i] > 0):
            alph_list = [alph for alph in self._alphas if 0 < alph and alph < self.C]
            if alph_list:
                j = heuristic(self._errors)  # TODO: min-max heap
                if self._take_step(i, j):
                    return True

                for j, alph in enumerate(np.shuffle(alph_list)):
                    if self._take_step(i, j):
                        return True

                for j, alph in enumerate(np.permute(range(n))):
                    if self._take_step(i, j):
                        return True

        return False

    @property
    def kernel(self, i, j):
        if i > j:
            i, j = j, i

        try:
            k_i = self._kernel[i]
        with KeyError:
            k_i = {}
            self._kernel[i] = k_i
        finally:
            try:
                return k_i[j]
            with KeyError:
                k_i[j] = torch.dot(self._X[i], self._X[j])
                return k_i[j]

    def fit(self, X, y):
        it = 0
        n = len(y)
        num_changed = 0
        examine_all = True
        self.coef_ = torch.matmul(X.T, self._alphas * y)
        self.intercept_ = np.random.randn(1)  # TODO: initial b
        self._errors = {}  # TODO: init min-max heap
        self._kernel = {}  # lazy computation with double-index indirection
        self._X = X
        self._y = y

        while it < self.max_iter and (num_changed > 0 or examine_all):
            it += 1
            num_changed = 0

            if examine_all:
                examine_all = False

                for i in range(n):
                    num_changed += self._examine_example(i)
            else:
                for i s.t. self._alphas[i] in (0, self.C):  # TODO: pythonise
                    num_changed += self._examine_example(i)

                if num_changed > 0:
                    examine_all = True

        del self._X
        del self._y

        self._num_iter = it

        return self

    def predict(self, X):
        f = torch.matmul(X, self.coef_) + self.intercept_
        return 2 * (f > 0) - 1

