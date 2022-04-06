"""
.. module:: utils
   :synopsis: This file contains ...
.. moduleauthor:: Angelo Ortiz <github.com/angelo-ortiz>
"""

import math
import numpy as np
import torch

from utils import layer_norm, euclidean_distances, dynamic_partition

def _init_centroids(X, n_clusters, x_sq_norms):
    n_samples, h, w, c = X.shape
    centres = torch.empty(
        n_clusters, h, w, c, dtype=X.dtype, device=X.device
    )

    n_local_trials = 2 + int(math.log(n_clusters))

    # Pick first center randomly and track index of point
    centre_id = np.random.randint(n_samples)
    # indices = torch.full(
    #     size=(n_clusters,), fill_value=-1, dtype=torch.int32, device=X.device
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
        candidate_ids = torch.searchsorted(
            torch.cumsum(closest_dist_sq.ravel(), dim=0),
            rand_vals
        )
        # XXX: numerical imprecision can result in a candidate_id out of range
        torch.clip(candidate_ids, max=closest_dist_sq.numel()-1, out=candidate_ids)

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

    return centres# , indices


def _is_same_clustering(labels1, labels2, n_clusters):
    mapping = torch.full(
        size=(n_clusters,), fill_value=-1, dtype=torch.int32, device=labels1.device
    )

    for i in range(len(labels1)):
        if mapping[labels1[i]] == -1:
            mapping[labels1[i]] = labels2[i]
        elif mapping[labels1[i]] != labels2[i]:
            return False
    return True


def _k_means_iter(X, x_sq_norms, centres, update_centres=True):
    n_clusters = len(centres)
    centres_new = torch.zeros_like(centres)
    labels = torch.full(
        size=(len(X),), fill_value=-1, dtype=torch.int64, device=X.device
    )
    centre_shift = torch.zeros(
        n_clusters, dtype=X.dtype, device=X.device
    )

    dist_sq = euclidean_distances(
        X, centres, X_norm_squared=x_sq_norms, squared=True
    )
    labels = torch.argmin(dist_sq, dim=1)
    X_part = dynamic_partition(X, labels, num_partitions=n_clusters)
    for c in range(n_clusters):
        centres_new[c] = torch.mean(X_part[c], dim=0)

    centre_shift = layer_norm(centres_new - centres, squared=False)

    return centres_new, labels, centre_shift


def _k_means_single(X, x_sq_norms, centres, max_iter=300, tol=1e-4):
    labels_old = torch.full(
        size=(len(X),), fill_value=-1, dtype=torch.int64, device=X.device
    )
    strict_convergence = False

    for i in range(max_iter):
        centres_new, labels, centre_shift = _k_means_iter(
            X,
            x_sq_norms,
            centres
        )

        centres = centres_new

        if torch.equal(labels, labels_old):
            # First check the labels for strict convergence.
            strict_convergence = True
            break
        else:
            # No strict convergence, check for tol-based convergence.
            centre_shift_tot = torch.square(centre_shift).sum()
            if centre_shift_tot <= tol:
                break

        labels_old[:] = labels

    if not strict_convergence:
        # rerun E-step so that predicted labels match cluster centers
        _, labels, _ = _k_means_iter(
            X,
            x_sq_norms,
            centres,
            update_centres=False
        )

    # Sum of squared distance between each sample and its assigned center
    inertia = layer_norm(X - centres[labels], squared=True).sum().item()

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
    x_sq_norms = layer_norm(X, squared=True)
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
              f'found less than n_clusters ({n_clusters})!', sep=' ')

    return best_centres, best_labels, best_n_iter
