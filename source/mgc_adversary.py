"""
Create the generator module
"""
import re
import numpy as np
import torch.nn as nn


class GeneratorNet(nn.Module):
    """ take a tensor with accuracies for each of the classes
    and predict a probability distribution for the classes that
    will maximize the loss of the classifier
    """

    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-arguments
    def __init__(
        self, 
        out_channels,  
        n_classes, 
        n_hidden_layers,
        n_hidden_features,
    ):
        super().__init__()

        # initialize some values
        self.out_channels = out_channels
        self.n_classes = n_classes
        self.img_size = img_size
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_features = n_hiden_features

        ####################################################
        ########             layers               ##########
        ####################################################
        
        # dense layers
        # initialize the first layer
        n_features_prev = self.n_classes
        n_features_out = self.n_hidden_features
        # stack linears
        for i in range(n_hidden_layers):
            setattr(
                self,
                "linear" + str(i),
                nn.Linear(
                    in_features=n_features,
                    out_features=n_features_out,
                ),
            )
            n_features_prev = n_features_out
            # add the output layer
            if i == self.n_hidden_layers-1:
                n_features_out = self.n_classes

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
