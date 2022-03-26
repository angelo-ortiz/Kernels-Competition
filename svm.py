"""
.. module:: svm
   :synopsis: This file contains a linear SVM solver.
.. moduleauthor:: Angelo Ortiz <github.com/angelo-ortiz>
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import layer_norm, extract_sq_patches, gaussian_window)

class LinearSVC:
    def __init__(self, n_classes, C=1., tol=1e-3, eps=1e-6, max_iter=-1):
        super().__init__()
        self.n_classes = n_classes
        self.C = C
        self.tol = tol  # margin tolerance
        self.eps = eps  # dual coefficients tolerance
        self.max_iter = np.inf if max_iter < 0 else max_iter
        self._kernel = {}  # lazy computation with double-index indirection

    @property
    def kernel(self, i, j):
        # exploit symmetry
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

    def _initialise_params(self, n_samples, n_features):
        # TODO: dual_coef_ necessarily on cuda?
        self.dual_coef_ = torch.zeros(self.n_classes, n_samples, device=self._device)
        self.coef_ = torch.zeros(self.n_classes, n_features, device=self._device)
        self.intercept_ = torch.zeros(self.n_classes, device=self._device)
        self.n_iter_ = 0

    def _non_bound_indices(self, c):
        non_bound = self.dual_coef_[c] > 0 & self.dual_coef_[c] < self.C
        return torch.nonzero(non_bound).squeeze()

    def _take_step(self, c, i, j):
        if i == j:
            return False

        s = self._y[i] * self._y[j]
        alph_i = self.dual_coef_[c, i]
        alph_j = self.dual_coef_[c, j]
        alph_gap = alph_j + s * alph_i
        L_j = max(0, alph_gap - self.C * (s+1)/2)
        H_j = min(self.C, alph_gap - self.C * (s-1)/2)
        if L_j == H_j:
            return False

        a_j = alph_j

        k_ii = self.kernel(i, i)
        k_ij = self.kernel(i, j)
        k_jj = self.kernel(j, j)
        eta = k_ii + k_jj - 2 * k_ij
        if eta > 0:
            a_j += self._y[j]*(self._errors[i] - self._errors[j])/eta
            a_j = np.clip(a_j, L_j, H_J)
        else:
            f_i = self._y[i]*(self._errors[i]-self._intercept_[c]) \
                - alph_i*k_ii - s*alph_j*k_ij
            f_j = self._y[j]*(self._errors[j]-self._intercept_[c]) \
                - s*alph_i*k_ij - alph_j*k_jj
            L_i = alph_i + s*(alph_j - L_j)
            H_i = alph_i + s*(alph_j - H_j)
            L_obj = L_i*f_i + L_j*f_j + (L_i**2)*k_ii/2 + (L_j**2)*k_jj/2 \
                + s*L_i*L_j*k_ij
            H_obj = H_i*f_i + H_j*f_j + (H_i**2)*k_ii/2 + (H_j**2)*k_jj/2 \
                + s*H_i*H_j*k_ij
            if L_obj < H_obj - self.eps:
                a_j = L_j
            elif L_obj > H_obj + self.eps:
                a_j = H_j

        gap_j = a_j - alph_j
        if abs(gap_j) < self.eps*(a_j + alph_j + self.eps):
            return False

        a_i = alph_i - s * gap_j

        # update the intercept
        intercept_i = self._errors[i] + self._y[j] * gap_j * (k_ij - k_ii)
        intercept_j = self._errors[j] + self._y[j] * gap_j * (k_jj - k_ij)
        if 0 < a_i and a_i < self.C:
           intercept_diff += intercept_i
        elif 0 < a_j and a_j < self.C:
            intercept_diff += intercept_j
        else:
            intercept_diff += (intercept_i + intercept_j) / 2
        self.intercept_[c] += intercept_diff

        # update the weight vector
        coef_diff = self._y[j] * gap_j * (self._X[j] - self._X[i])
        self.coef_[c] += coef_diff

        # update the error cache
        self._errors += intercept_diff + torch.matmul(self._X, coef_diff)

        self.dual_coef_[c, i] = a_i
        self.dual_coef_[c, j] = a_j

        return True


    def _examine_example(self, c, i):
        r_i = self._y[i] * self._errors[i]

        if (r_i < -self.tol and self.dual_coef_[c, i] < self.C) \
           or (r_i > self.tol and self.dual_coef_[c, i] > 0):
            if len(self._non_bound) > 1:
                # second-choice heuristic
                if self._errors[i] >= 0:
                    j = torch.argmin(self._errors[self._non_bound])
                else:
                    j = torch.argmax(self._errors[self._non_bound])
                if self._take_step(c, i, j):
                    return True

                non_bound_perm = torch.randperm(len(self._non_bound), device=self._device)
                for j in self._non_bound[non_bound_perm]:
                    if self._take_step(c, i, j):
                        return True

                for j in torch.randperm(len(self._y)):  # TODO: skip j = i?
                    if self._take_step(c, i, j):
                        return True

        return False

    def _fit_class_vs_rest(self, c, y):
        iter_ = 0
        num_changed = 0
        examine_all = True
        self._y = torch.where(y == c, 1, -1)
        self._errors = -self._y  # w_0 = 0, b_0 = 0
        self._non_bound = self._non_bound_indices(c)

        while iter_ < self.max_iter and (num_changed > 0 or examine_all):
            iter_ += 1
            num_changed = 0

            if examine_all:
                examine_all = False

                for i in range(len(y)):
                    num_changed += self._examine_example(i)
            else:
                for i in self._non_bound:
                    num_changed += self._examine_example(c, i)

                if num_changed > 0:
                    examine_all = True

        self.n_iter_ += iter_

        del self._y
        del self._errors
        self self._non_bound

        return self

    def fit(self, X, y):
        self._X = X
        self._device = X.device

        self._initialise_params(*X.shape)

        for c in range(self.n_classes):
            self._fit_class_vs_rest(c, y)

        del self._X
        del self._y

        return self

    def decision_function(self, X):
        return torch.matmul(X, self.coef_.T) + self.intercept_

    def predict(self, X):
        return torch.argmax(self.decision_function(X), dim=1)
