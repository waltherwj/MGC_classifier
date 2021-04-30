"""
Create the classifier module
"""

import re
import numpy as np
import torch.nn as nn


class MgcNet(nn.Module):
    """ take an MGC correlation map and predict
    the operations that could have taken to that
    map"""

    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-arguments
    def __init__(
        self, out_channels, num_conv_layers, n_classes, kernel=3, img_size=128
    ):
        super().__init__()

        # initialize some values
        self.out_channels = out_channels
        self.n_classes = n_classes
        self.img_size = img_size
        self.num_conv_layers = num_conv_layers

        # get the padding for same padding. Assumes stride=1
        pad = int(np.ceil((kernel - 1) / 2))

        ####################################################
        # layers
        # convolutional layers
        self.conv_input = nn.Conv2d(
            in_channels=1, out_channels=out_channels, kernel_size=kernel, padding=pad
        )
        self.successive_poolings = 0
        # account for the input convolutional layer too (-1)
        for i in range(num_conv_layers - 1):
            setattr(
                self,
                "conv" + str(i),
                nn.Conv2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel,
                    padding=1,
                ),
            )
            if img_size // (2 ** self.successive_poolings) > 32:
                setattr(
                    self,
                    "maxpool" + str(i),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                )
                self.successive_poolings += 1
            else:
                setattr(
                    self,
                    "maxpool" + str(i),
                    nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                )
        pooled_output_size = img_size // (2 ** self.successive_poolings)

        # linear layers
        in_features_linear = out_channels * pooled_output_size * pooled_output_size
        transition_features_linear = (
            n_classes + (out_channels * pooled_output_size * pooled_output_size) // 2
        )
        setattr(
            self,
            "linear" + str(0),
            nn.Linear(
                in_features=in_features_linear, out_features=transition_features_linear
            ),
        )
        setattr(
            self,
            "linear" + str(1),
            nn.Linear(in_features=transition_features_linear, out_features=n_classes),
        )

        # activation layers
        self.relu = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=1)
        self.define_sequence()

    def define_sequence(self):
        "set sequence to make it easier for the forward pass"
        self.sequence = []
        n_appended_convs = 0

        for name, seq_module in self.named_modules():
            number = []

            # set convolutional step
            if isinstance(seq_module, nn.Conv2d):
                n_appended_convs += 1
                self.sequence.append(seq_module)
                self.sequence.append(self.relu)
                number = re.findall(r"\d+", name)

            # check if list is not empty to set the maxpool layers
            if (len(number) > 0) and "conv" in name:
                # find the maxpool with corresponding number
                for maxpool_name, maxp_module in self.named_modules():
                    maxpool_number = re.findall(r"\d+", maxpool_name)
                    if (
                        isinstance(maxp_module, nn.MaxPool2d)
                        and maxpool_number == number
                    ):
                        self.sequence.append(maxp_module)

            # set the linear steps
            if (
                isinstance(seq_module, nn.Linear)
                and n_appended_convs == self.num_conv_layers
            ):
                self.sequence.append(seq_module)
                self.sequence.append(self.relu)

        # remove the last relu and change to softmax
        self.sequence.pop(-1)
        self.sequence.append(self.softmax)

    def forward(self, tensor):
        "forward pass of the MGC classifier"

        # control changing x to a view or not
        first_linear = True

        # iterate through the layers
        for layer in self.sequence:
            # if the first linear layer, change tensor to be in the correct shape

            if isinstance(layer, nn.Linear) and first_linear:
                pooled_output_size = self.img_size // (2 ** self.successive_poolings)
                tensor = tensor.view(
                    -1, self.out_channels * pooled_output_size * pooled_output_size
                )
                first_linear = False

            # apply the layer
            tensor = layer(tensor)
        return tensor
