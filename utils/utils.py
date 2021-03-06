import numpy as np
import random
import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "../"))


def seed_all(seed=42):
    torch.cuda.empty_cache()

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    random.seed(seed)

    np.random.seed(seed)


def seed_worker():
    worker_seed = 42
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(worker_seed)
    torch.cuda.manual_seed_all(worker_seed)

    np.random.seed(worker_seed)
    random.seed(worker_seed)


def weights_init(m):

    if isinstance(m, nn.Conv2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        torch.nn.init.zeros_(m.bias)
