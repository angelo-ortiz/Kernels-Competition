import timeit
import numpy as np
import pandas as pd
import torch
from layers import CKN

WIDTH = 32
HEIGHT = 32
RGB2GREY = [0.2989, 0.5870, 0.1140]

def to_grey(X):
    X = X.reshape(-1, 3, HEIGHT, WIDTH).transpose(0, 2, 3, 1)
    return np.expand_dims(X @ RGB2GREY, axis=-1)

def standardisation(X_train, X_test, to_grey=False):
    X = np.concatenate([X_train, X_test], axis=0)
    X = X.reshape(-1, 3, HEIGHT, WIDTH).transpose(0, 2, 3, 1)
    X -= X.mean(axis=(0, 1, 2))
    X /= X.std(axis=(0, 1, 2))
    if to_grey:
        X = np.expand_dims(X @ RGB2GREY, axis=-1)
    # X = X.transpose(0, 3, 1, 2)
    return X[:len(X_train)], X[len(X_train):]

def min_max_normalisation(X_train, X_test, to_grey=False):
    X = np.concatenate([X_train, X_test], axis=0)
    X = X.reshape(-1, 3, HEIGHT, WIDTH).transpose(0, 2, 3, 1)
    X -= X.min(axis=(0, 1, 2))
    X /= X.max(axis=(0, 1, 2))
    if to_grey:
        X = np.expand_dims(X @ RGB2GREY, axis=-1)
    # X = X.transpose(0, 3, 1, 2)
    return X[:len(X_train)], X[len(X_train):]

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

    return X[train_indices], y[train_indices], \
            X[val_indices], y[val_indices]

def main(train_maps_fn, test_maps_fn, y_train_fn):
    X_train = np.array(
        pd.read_csv('X_train.csv', header=None, sep=',',
                    usecols=range(3*HEIGHT*WIDTH), dtype=np.float32)
    )
    y_train = np.array(pd.read_csv('y_train.csv', header='infer', sep=','))[:, 1]  # discard the id column
    classes = np.unique(y_train)

    X_test = np.array(
        pd.read_csv('X_test.csv', header=None, sep=',',
                    usecols=range(3*HEIGHT*WIDTH), dtype=np.float32)
    )

    # X_train, X_test = min_max_normalisation(X_train, X_test)
    X_train = X_train.reshape(-1, 3, HEIGHT, WIDTH).transpose(0, 2, 3, 1)
    X_test = X_test.reshape(-1, 3, HEIGHT, WIDTH).transpose(0, 2, 3, 1)

    X_train, y_train, X_val, y_val = split_train_set(
        X_train,
        y_train,
        classes,
        val_size=0,
        seed=12345
    )

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'

    X_train = torch.from_numpy(X_train).to(device)
    y_train = torch.from_numpy(y_train).to(device)

    X_test = torch.from_numpy(X_test).to(device)

    print(f'Running on {device}')

    start = timeit.default_timer()
    model = CKN(
        out_filters=[50, 200],
        patch_sizes=[2, 2],
        subsample_factors=[2, 4],
        lrs=[1e-3, 1e-4],
        solver_samples=100000,
        batch_size=100,
        epochs=200
    )
    model.train(X_train)
    train_time = timeit.default_timer() - start

    start = timeit.default_timer()
    X_train_maps = model.forward(X_train)
    X_test_maps = model.forward(X_test)
    map_comp_time = timeit.default_timer() - start

    print(f'Saving maps...', end='')
    torch.save(X_train_maps, train_maps_fn)
    torch.save(X_test_maps, test_maps_fn)
    torch.save(y_train, y_train_fn)
    print('done!')

    print(f'Output map shape {X_train_maps.shape}')
    print(f'Training took {train_time:.4e} s')
    print(f'Map computation took {map_comp_time:.4e} s')

if __name__ == '__main__':
    main(train_maps_fn='x_train_maps.pt',
         test_maps_fn='x_test_maps.pt',
         y_train_fn='y_tensor.pt')
