"""
.. module:: utils
   :synopsis: This file contains ...
.. moduleauthor:: Angelo Ortiz <github.com/angelo-ortiz>
"""

import numpy as np
import torch

from utils import batch_norm, euclidean_distances

def _init_centroids(X, n_clusters, x_sq_norms):
    n_samples, h, w, c = X.shape
    centres = torch.empty(
        n_clusters, h, w, c, dtype=X.dtype, device=X.device
    )

    n_local_trials = 2 + int(np.log(n_clusters))

    # Pick first center randomly and track index of point
    centre_id = np.random.randint(n_samples)
    # indices = torch.full(
    #     size=n_clusters, fill_value=-1, dtype=torch.int32, device=X.device
    # )
    centres[0] = X[centre_id]
    # indices[0] = centre_id

    # Initialise list of closest distances and calculate current potential
    closest_dist_sq = euclidean_distances(
        centres[0].unsqueeze(0), X, Y_norm_squared=x_sq_norms, squared=True
    )
    current_pot = closest_dist_sq.sum()

    # Pick the remaining n_clusters-1 points
    for c in range(1, n_clusters):
        # Choose centre candidates by sampling with probability proportional
        # to the squared distance to the closest existing centre
        rand_vals = torch.rand(n_local_trials, device=X.device) * current_pot
        candidate_ids = torch.searchsorted(torch.cumsum(closest_dist_sq), rand_vals)
        # XXX: numerical imprecision can result in a candidate_id out of range
        torch.clip(candidate_ids, None, closest_dist_sq.numel - 1, out=candidate_ids)

        # Compute distances to center candidates
        distance_to_candidates = euclidean_distances(
            X[candidate_ids], X, Y_norm_squared=x_sq_norms, squared=True
        )

        # update closest distances squared and potential for each candidate
        torch.minimum(closest_dist_sq, distance_to_candidates, out=distance_to_candidates)
        candidates_pot = distance_to_candidates.sum(dim=1)

        # Decide which candidate is the best
        best_candidate = torch.argmin(candidates_pot).item()
        current_pot = candidates_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]

        # Permanently add best center candidate found in local tries
        centres[c] = X[best_candidate]
        # indices[c] = best_candidate

    return centers# , indices


def _is_same_clustering(labels1, labels2, n_clusters):
    mapping = torch.full(
        size=n_clusters, fill_value=-1, dtype=torch.int32, device=labels1.device
    )

    for i in range(len(labels1)):
        if mapping[labels1[i]] == -1:
            mapping[labels1[i]] = labels2[i]
        elif mapping[labels1[i]] != labels2[i]:
            return False
    return True

def _inertia(X, centres, labels):
    # TODO
    return inertia

def _k_means_iter(X, x_sq_norms, centres, update_centres=True):
    n_clusters = len(centres)
    centres_new = torch.zeros_like(centres)
    labels = torch.full(
        size=len(X), fill_value=-1, dtype=torch.int32, device=X.device
    )
    weight_in_clusters = torch.zeros(
        n_clusters, dtype=X.dtype, device=X.device
    )
    centre_shift = torch.zeros_like(weight_in_clusters)
    # TODO

    return centres_new, weight_in_clusters, labels, centre_shift


def _k_means_single(X, x_sq_norms, centres, max_iter=300, tol=1e-4):
    labels_old = None
    # labels.clone()
    strict_convergence = False

    for i in range(max_iter):
        centres_new, weight_in_clusters, labels, centre_shift = _k_means_iter(
            X,
            x_sq_norms,
            centres
        )

        centres, centres_new = centres_new, centres

        if labels_old is not None and torch.equal(labels, labels_old):
            # First check the labels for strict convergence.
            strict_convergence = True
            break
        else:
            # No strict convergence, check for tol-based convergence.
            centre_shift_tot = torch.square(centre_shift).sum()
            if center_shift_tot <= tol:
                break

        if labels_old is None:
            labels_old = labels.clone()
        else:
            labels_old[:] = labels

    if not strict_convergence:
        # rerun E-step so that predicted labels match cluster centers
        _, _, labels, _ = _k_means_iter(
            X,
            x_sq_norms,
            centres,
            update_centres=False
        )

    inertia = _inertia(X, centres, labels)

    return labels, inertia, centres, i + 1


def k_means(X, n_clusters, n_init=10, max_iter=300, tol=1e-4):
    """
    Parameters
    ----------
    X : Tensor[batch, h, w, c]
        The data points.
    n_clusters : int, > 0
        The number of clusters to form as well as the number of centroids
        to generate.
    n_init : init, default=10
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.
    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm for a
        single run.
    tol : float, default=1e-4
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.
    """
    x_squared_norms = batch_norms(X, squared=True)
    best_inertia, best_labels = None, None

    # subtract of mean of x for more accurate distance computations
    X_mean = X.mean(dim=0)
    X = X - X_mean  # avoid overwriting the given tensor

    for i in range(n_init):
        centres = _init_centroids(X, n_clusters, x_sq_norms)
        labels, inertia, centres, n_iter_ = _k_means_single(
            X,
            x_sq_norms,
            centres,
            max_iter=max_iter,
            tol=tol,
        )

        if best_inertia is None or (
                inertia < best_inertia
                and not _is_same_clustering(labels, best_labels, n_clusters)
        ):
            best_labels = labels
            best_centres = centres
            best_inertia = inertia
            best_n_iter = n_iter_

    best_centres += X_mean

    distinct_clusters = len(set(best_labels))
    if distinct_clusters < n_clusters:
        print(f'Warning: Number of distinct clusters ({distinct_clusters})',
              f'found less than n_clusters ({self.n_clusters})!')

    return best_centres, best_labels, best_n_iter
