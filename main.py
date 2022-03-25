"""
.. module:: main
   :synopsis: This file contains the entry point for the optim.
.. moduleauthor:: Angelo Ortiz <github.com/angelo-ortiz>
"""

import argparse

import numpy as np
import pandas as pd


import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import layer_norm, extract_sq_patches, gaussian_window
from layers import CKN
from svm import SVM

HEIGHT = 32
WIDTH = 32

RGB2GREY = [0.2989, 0.5870, 0.1140]

def preprocess(X, to_grey=False):
    X = X.reshape(-1, 3, HEIGHT, WIDTH).transpose(0, 2, 3, 1)
    X -= X.min(axis=(0, 1, 2))
    X /= X.max(axis=(0, 1, 2))
    if to_grey:
        return (X @ RGB2GREY).reshape(-1, HEIGHT, WIDTH, 1)
    return X

def split_train_set(X, y, classes, val_size=0.2, seed=None):
    rng = np.random.default_rng(seed)

    nb_examples = len(y) // len(classes)
    val_examples = int(val_size * nb_examples)

    class_indices = np.zeros((len(classes), nb_examples), dtype=int)
    for i, class_ in enumerate(classes):
        class_indices[i] = np.argwhere(y == class_).ravel()

    val_indices = rng.choice(
        class_indices,
        size=val_examples,
        replace=False,
        axis=1
    )

    train_indices = np.zeros((len(classes), nb_examples - val_examples), dtype=int)
    for i in range(len(classes)):
        train_indices[i] = np.setdiff1d(
            class_indices[i],
            val_indices[i],
            assume_unique=True
        )

    train_indices = rng.permutation(train_indices.ravel())
    val_indices = rng.permutation(val_indices.ravel())

    return X_train[train_indices], y_train[train_indices], \
            X_train[val_indices], y_train[val_indices]


def main(args):
    device = 'cpu' if args.gpu < 0 else f'cuda:{args.gpu}'

    ckn = CKN(
       out_filters=[12, 200],
       patch_sizes=[2, 2],
       subsample_factors=[2, 4],
       solver_samples=40000,
       batch_size=100,
       epochs=100
    )

    if args.mode == 'train':
        X_train = np.array(
            pd.read_csv(args.X_train, header=None, sep=',', usecols=range(3*HEIGHT*WIDTH))
        )

        # discard the id column
        y_train = np.array(pd.read_csv(args.y_train, header='infer', sep=','))[:, 1]
        classes = np.unique(y_train)

        X_train, y_train, X_val, y_val = split_train_set(
            X_train,
            y_train,
            classes,
            seed=args.val_seed
        )
        X_train = preprocess(X_train)
        X_train = torch.from_numpy(X_train).to(device)
        y_train = torch.from_numpy(y_train).to(device)

        X_val = preprocess(X_val)
        X_train = torch.from_numpy(X_val).to(device)
        y_val = torch.from_numpy(y_val).to(device)
        # torch.save(model.state_dict(), MODEL_STATE_FILE)

    # import model from file
    # model.load_state_dict(torch.load(MODEL_STATE_FILE, map_location=device))

    # test the model
    X_test = np.array(
        pd.read_csv(args.X_test, header=None, sep=',', usecols=range(3*HEIGHT*WIDTH))
    )
    X_test = preprocess(X_test)
    X_test = torch.from_numpy(X_test).to(device)

    # test(model, loss_fcn, device, test_dataloader)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',  choices=['train', 'test'], default='train')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU to use. Set -1 to use CPU.')
    parser.add_argument('--X_train', default='X_train.csv', help='The (relative) location of the training images.')
    parser.add_argument('--y_train', default='y_train.csv', help='The (relative) location of the training labels.')
    parser.add_argument('--X_test', default='X_test.csv', help='The (relative) location of the test images.')
    parser.add_argument('--val_seed', type=int, default=None, help='The seed for the validation-set extraction.')
    # parser.add_argument("--epochs", type=int, default=250)
    # parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    main(args)
