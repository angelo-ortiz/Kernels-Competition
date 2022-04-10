import timeit
import numpy as np
import pandas as pd
import torch
from svm import LinearSVC

COLUMNS = ['Prediction']

def main(train_maps_fn, test_maps_fn, y_train_fn):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    x_train_maps = torch.load(train_maps_fn, map_location=device)
    x_test_maps = torch.load(test_maps_fn, map_location=device)
    y_train = torch.load(y_train_fn, map_location=device)

    x_train_maps = x_train_maps.view(len(x_train_maps), -1)
    x_test_maps = x_test_maps.view(len(x_test_maps), -1)

    x_train_maps -= x_train_maps.mean(dim=0)
    x_train_maps /= torch.linalg.norm(x_train_maps, keepdim=True)

    x_test_maps -= x_test_maps.mean(dim=0)
    x_test_maps /= torch.linalg.norm(x_test_maps, keepdim=True)

    num_classes = len(torch.unique(y_train))
    svm = LinearSVC(n_classes=3, C=1., max_iter=10000, eps=1e-4)
    svm.fit(x_train_maps, y_train)

    train_preds = svm.predict(x_train_maps)
    acc = (train_preds == y_train).sum() / len(y_train)
    print(f'Train accuracy: {acc.item():.4f}')
    preds = svm.predict(x_test_maps)
    torch.save(preds, 'submission.pt')
    df = pd.DataFrame(preds.detach().cpu().numpy(),
                      index=np.arange(len(x_test_maps))+1,
                      columns=COLUMNS)
    df.index.name = 'Id'
    df.to_csv('Yte.csv')


if __name__ == '__main__':
    main(train_maps_fn='x_train_maps.pt',
         test_maps_fn='x_test_maps.pt',
         y_train_fn='y_tensor.pt')
