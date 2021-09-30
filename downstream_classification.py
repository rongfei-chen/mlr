import sys
import os
import numpy as np
import torch
from torch.autograd import Variable
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from collections import Counter


sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "../"))

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
model.load_state_dict(torch.load('output/ConvAE.pt'))

dataset_name = "iemocap"

dataset = dataloading.dataset_features(dataset_name)
train_loader, val_loader, test_loader, _ = dataloading.classification_dataloaders(dataset_name, dataset, 0)

x_train, y_train = get_representations(model, train_loader, device)
print("Representation dimension: {}".format(x_train.shape[1]))
x_val, y_val = get_representations(model, val_loader, device)
x_train = np.concatenate((x_train, x_val))
y_train = y_train + y_val
x_test, y_test = get_representations(model, test_loader, device)

print("---> Classification for IEMOCAP")

iemocap_emotions = ["Neutral", "Happy", "Sad", "Angry"]
y_train_init = y_train
y_test_init = y_test
iemocap_results = ""
for label in range(4):
    y_train = one_vs_all_labels(y_train_init, label)
    y_test = one_vs_all_labels(y_test_init, label)
    pipe = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(random_state=42, max_iter=10000))])

    pipe.fit(x_train, y_train)
    preds = pipe.predict(x_test)
    acc_2 = accuracy_score(y_test, preds)
    f1_binary = f1_score(y_test, preds, average='weighted')
    iemocap_results += "\n{}: (Acc2, binary F1_weighted) = ({}, {})\n".format(
        iemocap_emotions[label], round(acc_2 * 100, 2), round(f1_binary * 100, 2))

print(iemocap_results)

print("---> Classification for CMU-MOSI")

dataset_name = "cmumosi"
dataset = dataloading.dataset_features(dataset_name)

train_loader, val_loader, test_loader, _ = dataloading.classification_dataloaders(dataset_name, dataset, 1)

x_train, y_train = get_representations(model, train_loader, device)
x_val, y_val = get_representations(model, val_loader, device)
x_train = np.concatenate((x_train, x_val))
y_train = y_train + y_val
x_test, y_test = get_representations(model, test_loader, device)

print(Counter(y_train))
pipe = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(random_state=42, max_iter=10000))])

pipe.fit(x_train, y_train)
preds = pipe.predict(x_test)

preds = np.array(preds) - 3
y_test = np.array(y_test) - 3

acc_7 = accuracy_score(y_test, preds)

y_test = [1 if label >= 0 else 0 for label in y_test]
preds = [1 if pred >= 0 else 0 for pred in preds]
acc_2 = accuracy_score(y_test, preds)
f1_binary = f1_score(y_test, preds, average='weighted')


print("\n(Acc7, Acc2, binary F1_weighted) = ({}, {}, {})\n".format(
    round(acc_7 * 100, 2), round(acc_2 * 100, 2), round(f1_binary * 100, 2)))

print("---> Classification for CMU-MOSEI")

dataset_name = "cmumosei"
dataset = dataloading.dataset_features(dataset_name)

train_loader, val_loader, test_loader, _ = dataloading.classification_dataloaders(dataset_name, dataset, 1)
x_train, y_train = get_representations(model, train_loader, device)
x_val, y_val = get_representations(model, val_loader, device)
x_train = np.concatenate((x_train, x_val))
y_train = y_train + y_val
x_test, y_test = get_representations(model, test_loader, device)

print(Counter(y_train))
pipe = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(random_state=42, max_iter=10000))])

pipe.fit(x_train, y_train)
preds = pipe.predict(x_test)

preds = np.array(preds) - 3
y_test = np.array(y_test) - 3

acc_7 = accuracy_score(y_test, preds)

y_test = [1 if label >= 0 else 0 for label in y_test]
preds = [1 if pred >= 0 else 0 for pred in preds]
acc_2 = accuracy_score(y_test, preds)
f1_binary = f1_score(y_test, preds, average='weighted')
print("\n(Acc7, Acc2, binary F1_weighted) = ({}, {}, {})\n".format(
    round(acc_7 * 100, 2), round(acc_2 * 100, 2), round(f1_binary * 100, 2)))
