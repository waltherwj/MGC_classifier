import sys
from pathlib import Path
print(1)
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader


sys.path.append("\\DESKTOP-SFKRGC9\MGC_classifier\source")
import mgc_classifier as mgc_classifier
import dataset_loader as datasets

from IPython.display import clear_output

class train_and_validate_static:
    
    """
    class that implements training the network and
    outputting validation metrics
    """

    def __init__(
        self,
        model,
        datapath,
        criterion,
        optimizer,
        lrs,
        batch_size,
        minibatch_size,
        num_workers,
        scheduler,
    ):

        # choose gpu or cpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # initialize the model
        self.model = model(
            out_channels=1, num_conv_layers=5, n_classes=40, img_size=128
        ).to(self.device)
        self.datapath = datapath
        self.optimizer = optimizer(self.model.parameters(), lr=lrs[1])
        self.criterion = criterion.to(self.device)
        self.scheduler = scheduler(
            self.optimizer, base_lr=lrs[0], max_lr=lrs[2], cycle_momentum=False
        )

        # hyperparameters
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.num_workers = num_workers

        # initialize the logging variables
        self.training_loss = []
        self.validation_loss = []
        self.validation_accuracy = []
        self.class_accuracy = np.zeros(40)

        # training and validation dataloaders
        self.train_ratio = 0.99
        self.train_dataloader = None
        self.validation_dataloader = None

    def train(self, epochs=3):
        """
        train the network
        """

        # get the full dataset in the folder
        folder_dataset = datasets.PklDataset(self.datapath)
        print(len(folder_dataset))

        # split data into training and test

        train_data, validation_data = random_split(
            dataset=folder_dataset,
            lengths=[
                int(len(folder_dataset) * self.train_ratio),
                len(folder_dataset) - int(len(folder_dataset) * self.train_ratio),
            ],
        )
        print(len(train_data))

        # get the DataLoaders
        self.train_dataloader = DataLoader(
            dataset=train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=datasets.PklDataset.collate_fn,
        )
        self.validation_dataloader = DataLoader(
            dataset=validation_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=datasets.PklDataset.collate_fn,
        )

        for epoch in range(10):
            # initialize the training loss averaging list
            temp_loss = []
            # iterate through the samples
            ii = 0
            for i, batch_sample in enumerate(self.train_dataloader):
                minibatch = datasets.MinibatchDataset(data=batch_sample,)
                minibatch_dataloader = DataLoader(
                    dataset=minibatch,
                    batch_size=self.minibatch_size,
                    shuffle=True,
                    pin_memory=True,
                )
                for inputs, targets in minibatch_dataloader:

                    # send minibatch to gpu or cpu
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    # forward pass
                    predictions = self.model(inputs)

                    # loss
                    loss = self.criterion(predictions, targets)

                    # backward step
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    # zero out gradients for next step
                    self.optimizer.zero_grad()

                    temp_loss.append(loss.item())

                    print(f"\r{i} {ii}", end="")
                    ii += 1
                if i % 2 == 0:
                    self.validate()
                    self.training_loss.append(np.mean(temp_loss))
                if i % 2 == 0 and i != 0:
                    moving_window = -(15)
                    if len(self.training_loss) < moving_window:
                        moving_window = 0
                    clear_output(wait=True)
                    plt.plot(self.training_loss[moving_window:])
                    plt.plot(self.validation_loss[moving_window:])
                    plt.legend(["train", "val"])
                    plt.twinx().plot(self.validation_accuracy[moving_window:], "g")
                    plt.pause(0.0001)
                    plt.bar(
                        x=datasets.PklDataset(0).categories, height=self.class_accuracy
                    )
                    plt.xticks(rotation=90)
                    plt.pause(0.0001)
                if len(self.validation_loss) > 10:
                    if self.validation_loss[-1] < self.validation_loss[-2]:
                        torch.save(
                            self.model.state_dict(),
                            Path("../models/MgcAdvClassifier.pkl"),
                        )

    def validate(self):
        """
        validate the network predictions
        """
        with torch.no_grad():
            # initalize list to average the accuracy
            # and loss over the validation dataset
            loss_temp = []
            accuracy_temp = []
            prediction_class_sum = np.zeros(40)
            class_accuracy_temp = []
            for i, batch_sample in enumerate(self.validation_dataloader):
                minibatch = datasets.MinibatchDataset(data=batch_sample,)
                minibatch_dataloader = DataLoader(
                    dataset=minibatch,
                    batch_size=self.minibatch_size,
                    shuffle=True,
                    pin_memory=True,
                )
                for inputs, targets in minibatch_dataloader:

                    # send minibatch to gpu or cpu
                    inputs = inputs.to(self.device)
                    targets = targets

                    # run prediction on a subset of the data
                    predictions = self.model(inputs)
                    predictions = predictions.cpu()

                    # get validation loss
                    loss_temp.append(
                        self.criterion(predictions, targets).numpy().item()
                    )

                    # get validation accuracy
                    for j in range(predictions.shape[0]):
                        tgts = targets[j, :]
                        preds = predictions[j, :]
                        k = int(torch.sum(tgts))
                        prediction_topk = torch.topk(preds, k=k).indices
                        prediction_classes = torch.zeros_like(preds)
                        prediction_classes[prediction_topk] = 1.0
                        number_correct_classes = torch.sum(
                            torch.logical_and(tgts, prediction_classes)
                        )
                        accuracy_temp.append(
                            (number_correct_classes / k).numpy().item()
                        )
                        class_accuracy_temp.append(
                            torch.logical_and(tgts, prediction_classes).float().numpy()
                        )
                if i >= 1:
                    break
            class_accuracy_temp = np.array(class_accuracy_temp)
            self.class_accuracy = (
                np.sum(class_accuracy_temp, axis=0) / class_accuracy_temp.shape[0]
            )
            self.validation_accuracy.append(np.mean(accuracy_temp))
            self.validation_loss.append(np.mean(loss_temp))


test_model = mgc_classifier.MgcNet

root_path = Path("../data")
train_val_object = train_and_validate_static(
    model=mgc_classifier.MgcNet,
    datapath=root_path,
    criterion=torch.nn.MultiLabelSoftMarginLoss(),
    optimizer=torch.optim.Adam,
    scheduler=torch.optim.lr_scheduler.CyclicLR,
    lrs=[1e-8, 1e-5, 1e-5],
    batch_size=10,  # just how much data can be loaded into memory at one time
    minibatch_size=5,  # what actually controls the batching size for training
    num_workers=2,
)

train_val_object.train()
