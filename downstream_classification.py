import sys
import os
import numpy as np
import torch
from torch.autograd import Variable
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from collections import Counter


sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "../"))

import utils.utils as utils
import utils.dataloading as dataloading
import utils.architectures as architectures


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

train_loader, val_loader, test_loader = dataloading.classification_dataloaders("cmumosi")

x_train, y_train = get_representations(model, train_loader, device)
x_val, y_val = get_representations(model, val_loader, device)
x_train = np.concatenate((x_train, x_val))
y_train = y_train + y_val
x_test, y_test = get_representations(model, test_loader, device)

print(Counter(y_train))
pipe = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression())])

pipe.fit(x_train, y_train)
preds = pipe.predict(x_test)

print(classification_report(y_test, preds))