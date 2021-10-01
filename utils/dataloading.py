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


class ClassificationDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


def stack_features(*features_stats):
    return np.dstack([*features_stats])


def drop_inf(x_train, x_val, x_test):
    for idx in range(x_train.shape[2]):
        if x_train[:, :, idx].min() == -inf:
            tmp = x_train[:, :, idx]
            tmp[tmp == -inf] = 0
            minimum = tmp.min()

            x_train[:, :, idx][x_train[:, :, idx] == -inf] = minimum
            x_val[:, :, idx][x_val[:, :, idx] == -inf] = minimum
            x_test[:, :, idx][x_test[:, :, idx] == -inf] = minimum

    return x_train, x_val, x_test


def scale_2d_min_max(x, maximum=None, minimum=None):

    if maximum is None and minimum is None:
        minimum = x.min(axis=(0, 1), keepdims=True)
        maximum = x.max(axis=(0, 1), keepdims=True)
    x = (x - minimum) / (maximum - minimum + 1e-10)

    return x, maximum, minimum


def scale_2d_standard(x, mu=None, std=None):
    if mu is None and std is None:
        mu = x.mean(axis=(0, 1), keepdims=True)
        std = x.std(axis=(0, 1), keepdims=True)
    x = (x - mu) / (std + 1e-10)

    return x, mu, std


def dataset_features(dataset_name):
    if dataset_name == "cmumosei" or dataset_name == "cmumosi":
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

    x_train, x_val, x_test = drop_inf(x_train, x_val, x_test)

    x_train, mu, std = scale_2d_standard(x_train)
    x_val, _, _ = scale_2d_standard(x_val, mu, std)
    x_test, _, _ = scale_2d_standard(x_test, mu, std)

    x_train, maximum, minimum = scale_2d_min_max(x_train)
    x_val, _, _ = scale_2d_min_max(x_val, maximum, minimum)
    x_test, _, _ = scale_2d_min_max(x_test, maximum, minimum)

    return x_train, y_train, x_val, y_val, x_test, y_test


def representation_dataloaders(test_dataset=None):
    torch.cuda.empty_cache()
    seed = 42
    utils.seed_all(seed)

    if test_dataset is None:
        x_train, y_train, x_val, y_val, x_test, y_test = dataset_features("cmumosei")
        """
        x_train_tmp, y_train_tmp, x_val_tmp, y_val_tmp, x_test_tmp, y_test_tmp = dataset_features("cmumosi")
        x_train = np.concatenate((x_train, x_train_tmp))
        x_val = np.concatenate((x_val, x_val_tmp))
        x_test = np.concatenate((x_test, x_test_tmp))
        """
        x_train_tmp, y_train_tmp, x_val_tmp, y_val_tmp, x_test_tmp, y_test_tmp = dataset_features("iemocap")
        x_train = np.concatenate((x_train, x_train_tmp))
        x_val = np.concatenate((x_val, x_val_tmp))
        x_test = np.concatenate((x_test, x_test_tmp))

    else:
        x_train, y_train, x_val, y_val, x_test, y_test = dataset_features("cmumosei")
        if test_dataset != "cmumosei":
            x_train = np.concatenate((x_train, x_test))
        """
        x_train_tmp, y_train_tmp, x_val_tmp, y_val_tmp, x_test_tmp, y_test_tmp = dataset_features("cmumosi")
        if test_dataset != "cmumosi":
            x_train = np.concatenate((x_train, x_train_tmp))
            x_train = np.concatenate((x_train, x_test_tmp))
            x_val = np.concatenate((x_val, x_val_tmp))
        else:
            x_train = np.concatenate((x_train, x_train_tmp))
            x_val = np.concatenate((x_val, x_val_tmp))
            x_test = np.concatenate((x_test, x_test_tmp))
        """
        x_train_tmp, y_train_tmp, x_val_tmp, y_val_tmp, x_test_tmp, y_test_tmp = dataset_features("iemocap")

        if test_dataset != "iemocap":
            x_train = np.concatenate((x_train, x_train_tmp))
            x_train = np.concatenate((x_train, x_test_tmp))
            x_val = np.concatenate((x_val, x_val_tmp))
        else:
            x_train = np.concatenate((x_train, x_train_tmp))
            x_val = np.concatenate((x_val, x_val_tmp))
            x_test = np.concatenate((x_test, x_test_tmp))

    indices_to_keep = list(range(74)) + list(range(-300, 0))

    train_set = RepresentationDataset(x_train[:, :, indices_to_keep])
    val_set = RepresentationDataset(x_val[:, :, indices_to_keep])
    test_set = RepresentationDataset(x_test[:, :, indices_to_keep])

    batch_size = 32

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, worker_init_fn=utils.seed_worker)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, worker_init_fn=utils.seed_worker)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, worker_init_fn=utils.seed_worker)

    return train_loader, val_loader, test_loader, batch_size


def cmumosei_round(a):
    if a < -2:
        res = -3
    if -2 <= a and a < -1:
        res = -2
    if -1 <= a and a < 0:
        res = -1
    if 0 <= a and a <= 0:
        res = 0
    if 0 < a and a <= 1:
        res = 1
    if 1 < a and a <= 2:
        res = 2
    if a > 2:
        res = 3
    return res


def sentiment_labels(a, dataset_name):
    if dataset_name == "cmumosi":
        reg_label = a[0][0]
    else:
        reg_label = a
    label_7 = cmumosei_round(a)
    if label_7 >= 0:
        label_2 = 1
    else:
        label_2 = 0
    return label_2, label_7 + 3, reg_label


def iemocap_label(classes_2d):
    for idx, label in enumerate(classes_2d):
        if label[1] == 1:
            return idx


def classification_dataloaders(dataset_name, dataset, idx=0):

    x_train, y_train, x_val, y_val, x_test, y_test = dataset

    if dataset_name == "cmumosi":
        y_train = [sentiment_labels(y, dataset_name)[idx] for y in y_train]
        y_val = [sentiment_labels(y, dataset_name)[idx] for y in y_val]
        y_test = [sentiment_labels(y, dataset_name)[idx] for y in y_test]
    elif dataset_name == "cmumosei":

        y_train = [sentiment_labels(y[0], dataset_name)[idx] for y in y_train.squeeze()]
        y_val = [sentiment_labels(y[0], dataset_name)[idx] for y in y_val.squeeze()]
        y_test = [sentiment_labels(y[0], dataset_name)[idx]for y in y_test.squeeze()]
    elif dataset_name == "iemocap":
        y_train = [iemocap_label(y) for y in y_train]
        y_val = [iemocap_label(y) for y in y_val]
        y_test = [iemocap_label(y) for y in y_test]

    train_set = ClassificationDataset(x_train, y_train)
    val_set = ClassificationDataset(x_val, y_val)
    test_set = ClassificationDataset(x_test, y_test)

    batch_size = 32

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, worker_init_fn=utils.seed_worker)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, worker_init_fn=utils.seed_worker)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, worker_init_fn=utils.seed_worker)

    return train_loader, val_loader, test_loader, batch_size
