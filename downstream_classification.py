import sys
import os
import numpy as np
import torch
from torch.autograd import Variable
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from collections import Counter


sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "../"))

import utils.utils as utils
import utils.dataloading as dataloading
import utils.architectures as architectures


def one_vs_all_labels(labels, positive_label):
    return [1 if label == positive_label else 0 for label in labels]


def get_representations(model, dataloader, device):
    model.eval()

    pred = []
    label = []
    for x, y in dataloader:
        x = Variable(x).float().to(device)
        x = x.unsqueeze(1)
        with torch.set_grad_enabled(False):
            # ===================forward=====================
            output, representation = model(x)
            pred.append(torch.squeeze(representation.float()).detach().cpu().numpy())
            label.append(y.detach().cpu().numpy())

    labels = [item for sublist in label for item in sublist]
    representations = [item for sublist in pred for item in sublist]

    return np.array(representations), labels


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print("Running on: {}".format(device))

model = architectures.ConvAutoEncoder(20, 409).to(device)
model.load_state_dict(torch.load('output/ConvAE_all.pt'))

dataset_name = "iemocap"

dataset = dataloading.dataset_features(dataset_name)
train_loader, val_loader, test_loader = dataloading.classification_dataloaders(dataset_name, dataset, 0)

x_train, y_train = get_representations(model, train_loader, device)
x_val, y_val = get_representations(model, val_loader, device)
x_train = np.concatenate((x_train, x_val))
y_train = y_train + y_val
x_test, y_test = get_representations(model, test_loader, device)

print("---> Classification for IEMOCAP")

iemocap_emotions = ["Neutral", "Happy", "Sad", "Angry"]
y_train_init = y_train
y_test_init = y_test
for label in range(4):
    y_train = one_vs_all_labels(y_train_init, label)
    y_test = one_vs_all_labels(y_test_init, label)
    print("       -------------------Label {}-------------------".format(iemocap_emotions[label]))
    print(Counter(y_train))
    pipe = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(random_state=42))])

    pipe.fit(x_train, y_train)
    preds = pipe.predict(x_test)

    print(classification_report(y_test, preds))

print("---> Classification for CMU-MOSI")

dataset_name = "cmumosi"
dataset = dataloading.dataset_features(dataset_name)

train_loader, val_loader, test_loader = dataloading.classification_dataloaders(dataset_name, dataset, 0)

x_train, y_train = get_representations(model, train_loader, device)
x_val, y_val = get_representations(model, val_loader, device)
x_train = np.concatenate((x_train, x_val))
y_train = y_train + y_val
x_test, y_test = get_representations(model, test_loader, device)

print(Counter(y_train))
pipe = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(random_state=42))])

pipe.fit(x_train, y_train)
preds = pipe.predict(x_test)

print(classification_report(y_test, preds))

print("---> Classification for CMU-MOSEI")

dataset_name = "cmumosei"
dataset = dataloading.dataset_features(dataset_name)

train_loader, val_loader, test_loader = dataloading.classification_dataloaders(dataset_name, dataset, 0)
x_train, y_train = get_representations(model, train_loader, device)
x_val, y_val = get_representations(model, val_loader, device)
x_train = np.concatenate((x_train, x_val))
y_train = y_train + y_val
x_test, y_test = get_representations(model, test_loader, device)

print(Counter(y_train))
pipe = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(random_state=42))])

pipe.fit(x_train, y_train)
preds = pipe.predict(x_test)

print(classification_report(y_test, preds))
