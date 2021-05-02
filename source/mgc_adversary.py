"""
Create the generator module
"""
import torch.nn as nn


class GeneratorNet(nn.Module):
    """ take a tensor with accuracies for each of the classes
    and predict a probability distribution for the classes that
    will maximize the loss of the classifier
    """

    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-arguments
    def __init__(
        self, n_classes, n_hidden_layers, n_hidden_features,
    ):
        super().__init__()

        # initialize some values
        self.n_classes = n_classes
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_features = n_hidden_features
        self.sequence = []

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
                nn.Linear(in_features=n_features_prev, out_features=n_features_out,),
            )
            n_features_prev = n_features_out
            # add the output layer
            print(i)
            if i + 2 == self.n_hidden_layers:
                n_features_out = self.n_classes

        # activation layers
        self.relu = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=1)
        self.define_sequence()

    def define_sequence(self):
        "set sequence to make it easier for the forward pass"

        for _, seq_module in self.named_modules():

            # set linear layer
            if isinstance(seq_module, nn.Linear):
                self.sequence.append(seq_module)
                self.sequence.append(self.relu)

        # remove the last relu and change to softmax
        self.sequence.pop(-1)
        self.sequence.append(self.softmax)

    def forward(self, tensor):
        "forward pass of the MGC classifier"

        # iterate through the layers
        for layer in self.sequence:
            # apply the layer
            tensor = layer(tensor)
        return tensor
