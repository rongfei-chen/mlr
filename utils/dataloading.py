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


def stack_features(*features_stats):
    return np.dstack([*features_stats])


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


def scale_by_modality(x_train, x_val, x_test, down, up):
    x_train[:, :, down:up], maximum, minimum = scale(x_train[:, :, down:up])
    x_val[:, :, down:up] = scale(x_val[:, :, down:up], maximum, minimum)
    x_test[:, :, down:up] = scale(x_test[:, :, down:up], maximum, minimum)

    return x_train, x_val, x_test


def dataset_features(dataset_name):
    if dataset_name == "cmumosei":
        seq_len = 20
        x_train, y_train, x_val, y_val, x_test, y_test = get_feature_matrix(dataset_name, seq_len)

    elif dataset_name == "iemocap":
        with open(r"data/iemocap_seq_20.pkl", "rb") as input_file:
            iemocap_data = pickle.load(input_file)
        y_train = iemocap_data["train"]["labels"]
        y_val = iemocap_data["valid"]["labels"]
        y_test = iemocap_data["test"]["labels"]
        x_train = stack_features(
            iemocap_data["train"]["audio"], iemocap_data["train"]["vision"], iemocap_data["train"]["text"])
        x_val = stack_features(
            iemocap_data["valid"]["audio"], iemocap_data["valid"]["vision"], iemocap_data["valid"]["text"])
        x_test = stack_features(
            iemocap_data["test"]["audio"], iemocap_data["test"]["vision"], iemocap_data["test"]["text"])

    x_train, x_val, x_test = scale_by_modality(x_train, x_val, x_test, 0, 74)
    x_train, x_val, x_test = scale_by_modality(x_train, x_val, x_test, 74, 109)
    x_train, x_val, x_test = scale_by_modality(x_train, x_val, x_test, 109, 409)
    return x_train, y_train, x_val, y_val, x_test, y_test


def dataloaders():
    torch.cuda.empty_cache()
    seed = 42
    utils.seed_all(seed)

    x_train, y_train, x_val, y_val, x_test, y_test = dataset_features("cmumosei")
    x_train_tmp, y_train_tmp, x_val_tmp, y_val_tmp, x_test_tmp, y_test_tmp = dataset_features("iemocap")
    x_train = np.concatenate((x_train, x_train_tmp))
    x_val = np.concatenate((x_val, x_val_tmp))
    x_test = np.concatenate((x_test, x_test_tmp))

    train_set = RepresentationDataset(x_train)
    val_set = RepresentationDataset(x_val)
    test_set = RepresentationDataset(x_test)

    batch_size = 32

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, worker_init_fn=utils.seed_worker)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, worker_init_fn=utils.seed_worker)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, worker_init_fn=utils.seed_worker)

    return train_loader, val_loader, test_loader, batch_size


train_loader, val_loader, test_loader, batch_size = dataloaders()
