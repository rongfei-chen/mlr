import sys
import os
import yaml
from tqdm import tqdm
import numpy as np
import random
import torch
from torch.autograd import Variable
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score

import pickle

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "../"))

import utils.utils as utils
import utils.dataloading as dataloading
import utils.architectures as architectures


class EpochTrain:
    """
    Class that implements epoch train and validation procedures.
    """
    def __init__(self, net, batch_size, train_loader, val_loader, task,
                 optimizer, scheduler, criterion, device):
        self.net = net
        self.batch_size = batch_size
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.task = task
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.train_epoch_losses = []
        self.val_epoch_losses = []

    def train(self, epoch):
        self.net.train()
        running_loss = 0.0

        with tqdm(self.train_loader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch + 1}")
            self.net.train()

            self.scheduler.step(epoch)
            running_loss = 0
            if self.task == "representation_learning":
                for it, x in enumerate(tepoch):
                    x = Variable(x).float().to(self.device)
                    x = x.unsqueeze(1)

                    with torch.set_grad_enabled(True):
                        # ===================forward=====================
                        output, representation = self.net(x)
                        loss = self.criterion(output, x)
                        # ===================backward====================
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                    running_loss += loss.item() * x.size(0)

                    tepoch.set_postfix(batch_loss=loss.item())
            else:
                for it, (x, y) in enumerate(tepoch):
                    x = Variable(x).float().to(self.device)
                    y = y.to(self.device)
                    x = x.unsqueeze(1)

                    with torch.set_grad_enabled(True):
                        # ===================forward=====================
                        output = self.net(x)
                        loss = self.criterion(output, y)
                        # ===================backward====================
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                    running_loss += loss.item() * x.size(0)

                    tepoch.set_postfix(batch_loss=loss.item())

            # ===================log========================
            epoch_loss = running_loss / (len(self.train_loader) * self.batch_size)
            epoch_score = self.validate()
            print("Epoch score: {}".format(epoch_score))

            epoch_loss = running_loss / (len(self.train_loader) * self.batch_size)

        return epoch_loss, epoch_score

    def validate(self):
        self.net.eval()

        pred = []
        actual = []
        running_loss = 0
        if self.task == "representation_learning":

            for x in self.val_loader:
                x = Variable(x).float().to(self.device)
                x = x.unsqueeze(1)
                with torch.set_grad_enabled(False):
                    # ===================forward=====================
                    output, representation = self.net(x)
                    loss = self.criterion(output, x)
                    pred.append(torch.squeeze(output.float()).detach().cpu().numpy())
                    actual.append(x.float().detach().cpu().numpy())
                running_loss += loss.item() * x.size(0)

            #actuals = [item for sublist in actual for item in sublist]
            #preds = [item for sublist in pred for item in sublist]

            score = running_loss / (len(self.val_loader) * self.batch_size)

        else:

            for x, y in self.val_loader:
                x = Variable(x).float().to(self.device)
                x = x.unsqueeze(1)
                y = y.to(self.device)

                with torch.set_grad_enabled(False):
                    # ===================forward=====================
                    output = self.net(x)
                    # TODO: possible argmax
                    loss = self.criterion(output, y)
                    pred.append(torch.squeeze(output.float()).detach().cpu().numpy())
                    actual.append(y.float().detach().cpu().numpy())
                running_loss += loss.item() * x.size(0)

            actuals = [item for sublist in actual for item in sublist]
            preds = [np.argmax(item) for sublist in pred for item in sublist]
            score = f1_score(actuals, preds, average="macro")
        #epoch_loss = running_loss / (len(self.val_loader) * self.batch_size)

        return score


def train_and_validate(task="representation_learning", dataset_name="iemocap"):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print("Running on: {}".format(device))

    print("--> Training Convolutional Auto-Encoder")

    utils.seed_all()
    if task == "representation_learning":
        best_score = 100
        comparison = 1
        train_loader, val_loader, test_loader, batch_size = dataloading.representation_dataloaders()

        model = architectures.ConvAutoEncoder(20, 409).to(device)
        criterion = nn.MSELoss()
        metric = "representation"
        model.apply(utils.weights_init)

    else:
        best_score = 0
        comparison = -1
        dataset = dataloading.dataset_features(dataset_name)

        train_loader, val_loader, test_loader, batch_size = dataloading.classification_dataloaders(
            dataset_name, dataset, 0)

        model = architectures.CNN(20, 409).to(device)
        criterion = torch.nn.CrossEntropyLoss()
        metric = "classification"

    print(model)

    train_params = {'lr': 2e-2, 'max_epochs': 100,
                    'batch_size': batch_size,
                    'device': device, 'print_every': 5}

    optimizer = torch.optim.Adam(
        model.parameters(), lr=train_params['lr'], weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', verbose=True)

    epoch_trainer = EpochTrain(
        model, train_params['batch_size'], train_loader, val_loader, task,
        optimizer, scheduler, criterion, train_params['device'])

    best_epoch = 1
    early_stop_counter = 0
    for epoch in range(train_params['max_epochs']):

        if early_stop_counter > 10:
            print('Best epoch was : Epoch {}'.format(best_epoch))
            break

        epoch_loss, epoch_score = epoch_trainer.train(epoch)

        if comparison * epoch_score < comparison * best_score:
            early_stop_counter = 0
            best_epoch = epoch + 1
            best_score = epoch_score
            torch.save(model.state_dict(), 'output/ConvAE_Checkpoint.pt')
        else:
            early_stop_counter += 1
    model = architectures.ConvAutoEncoder(20, 409).to(device)
    model.load_state_dict(torch.load('output/ConvAE_Checkpoint.pt'))
    torch.save(model.state_dict(), 'output/ConvAE.pt')
    epoch_trainer.net = model
    epoch_score = epoch_trainer.validate()
    print("Best score: {}".format(epoch_score))


if __name__ == "__main__":

    train_and_validate(task="representation_learning")

