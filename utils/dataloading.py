import os
import sys
import yaml
from tqdm import tqdm
import numpy as np
from numpy import inf
import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import pickle

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "../"))

from MultimodalSDK_loader.process_dataset import get_feature_matrix
import utils.utils as utils


class RepresentationDataset(Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index]


def scale(x, maximum=None, minimum=None):

    if maximum is None and minimum is None:
        maximum = x.max()
        minimum = x.min()
        if minimum == -inf:
            tmp = x
            tmp[tmp == -inf] = 0
            minimum = tmp.min()
            x[x == -inf] = minimum
        return (x - minimum) / (maximum - minimum), maximum, minimum
    else:
        if x.min() == -inf:
            x[x == -inf] = minimum
        return (x - minimum) / (maximum - minimum)


def scale_by_modality(x_train, x_val, x_test, up, down):
    x_train[:, :, up:down], maximum, minimum = scale(x_train[:, :, up:down])
    x_val[:, :, up:down] = scale(x_val[:, :, up:down], maximum, minimum)
    x_test[:, :, up:down] = scale(x_test[:, :, up:down], maximum, minimum)

    return x_train, x_val, x_test


def dataloaders():
    torch.cuda.empty_cache()
    seed = 42
    utils.seed_all(seed)

    dataset_name = "cmumosei"
    seq_len = 20
    x_train, y_train, x_val, y_val, x_test, y_test = get_feature_matrix(dataset_name, seq_len)

    x_train, x_val, x_test = scale_by_modality(x_train, x_val, x_test, 0, 74)
    x_train, x_val, x_test = scale_by_modality(x_train, x_val, x_test, 74, 109)
    x_train, x_val, x_test = scale_by_modality(x_train, x_val, x_test, 109, 409)

    # TODO: scaling

    train_set = RepresentationDataset(x_train)
    val_set = RepresentationDataset(x_val)
    test_set = RepresentationDataset(x_test)

    batch_size = 32

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, worker_init_fn=utils.seed_worker)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, worker_init_fn=utils.seed_worker)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, worker_init_fn=utils.seed_worker)

    return train_loader, val_loader, test_loader, batch_size


train_loader, val_loader, test_loader, batch_size = dataloaders()
