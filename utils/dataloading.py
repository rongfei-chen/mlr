import os
import sys
import yaml
from tqdm import tqdm
import numpy as np
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


def dataloaders():
    torch.cuda.empty_cache()
    seed = 42
    utils.seed_all(seed)

    dataset_name = "cmumosi"
    seq_len = 20
    x_train, y_train, x_val, y_val, x_test, y_test = get_feature_matrix(dataset_name, seq_len)

    # TODO: scaling

    train_set = RepresentationDataset(x_train)
    val_set = RepresentationDataset(x_val)
    test_set = RepresentationDataset(x_test)

    batch_size = 32

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, worker_init_fn=utils.seed_worker)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, worker_init_fn=utils.seed_worker)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, worker_init_fn=utils.seed_worker)

    """
    x = feature_stats

    batch_size = 128

    healthy = [x for idx, x in enumerate(x) if labels[idx] == 0]
    healthy_labels = np.zeros(len(healthy))

    damaged = [x for idx, x in enumerate(x) if labels[idx] == 1]
    damaged_labels = np.ones(len(damaged))

    x_train = np.array(healthy[:-len(damaged_labels)])

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)

    y_train = healthy_labels[:-len(damaged_labels)]
    x_val = np.concatenate([damaged, healthy[-len(damaged_labels):]])
    x_val = scaler.transform(x_val)
    y_val = np.concatenate([damaged_labels, healthy_labels[-len(damaged_labels):]])

    train_set = AEDataset(x_train, y_train)
    val_set = AEDataset(x_val, y_val)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, worker_init_fn=utils.seed_worker)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, worker_init_fn=utils.seed_worker)

    return train_loader, val_loader, batch_size, y_val, scaler
    """
    return train_loader, val_loader, test_loader, batch_size


train_loader, val_loader, test_loader, batch_size = dataloaders()
