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
    def __init__(self, net, batch_size, train_loader, val_loader,
                 optimizer, scheduler, criterion, device):
        self.net = net
        self.batch_size = batch_size
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.train_epoch_losses = []
        self.val_epoch_losses = []

    def train(self, epoch):
        self.net.train()
        self.scheduler.step(epoch)
        running_loss = 0.0

        with tqdm(self.train_loader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch + 1}")
            self.net.train()

            self.scheduler.step(epoch)
            running_loss = 0
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

            # ===================log========================
            epoch_loss = running_loss / (len(self.train_loader) * self.batch_size)
            epoch_score = self.validate()
            print("Epoch score: {}".format(epoch_score))

            epoch_loss = running_loss / (len(self.train_loader) * self.batch_size)

        return epoch_loss, epoch_score

    def validate(self, print_results=False):
        self.net.eval()

        pred = []
        actual = []
        running_loss = 0
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

        actuals = [item for sublist in actual for item in sublist]
        preds = [item for sublist in pred for item in sublist]

            # ===================log========================
        epoch_loss = running_loss / (len(self.val_loader) * self.batch_size)

        return epoch_loss


def train_and_validate():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print("Running on: {}".format(device))

    print("--> Training Convolutional Auto-Encoder")

    utils.seed_all()

    train_loader, val_loader, test_loader, batch_size = dataloading.dataloaders()

    model = architectures.ConvAutoEncoder(20, 409).to(device)
    print(model)
    model.apply(utils.weights_init)

    train_params = {'lr': 2e-2, 'max_epochs': 100,
                    'batch_size': batch_size,
                    'device': device, 'print_every': 5}

    optimizer = torch.optim.Adam(
        model.parameters(), lr=train_params['lr'], weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', verbose=True)
    criterion = nn.MSELoss()

    epoch_trainer = EpochTrain(
        model, train_params['batch_size'], train_loader, val_loader,
        optimizer, scheduler, criterion, train_params['device'])

    best_score = 100
    best_epoch = 1
    early_stop_counter = 0
    for epoch in range(train_params['max_epochs']):

        if early_stop_counter > 10:
            print('Best epoch was : Epoch {}'.format(best_epoch))
            break

        epoch_loss, epoch_score = epoch_trainer.train(epoch)

        if epoch_score < best_score:
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
    epoch_score = epoch_trainer.validate(print_results=True)
    print("Best score: {}".format(epoch_score))


if __name__ == "__main__":

    train_and_validate()

