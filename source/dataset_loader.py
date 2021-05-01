"""
the dataset classes to load the data into the neural network
"""

import glob
import pickle
from pathlib import Path

import numpy as np
import torch


from torch.utils.data import Dataset


class PklDataset(Dataset):
    """Get a pkled file from the folder and return
    the formatted dataset that corresponds to it"""

    # pylint: disable=no-member
    # pylint: disable=too-many-locals
    def __init__(self, root_dir):
        """
        Args:
            root_dir (string): Directory with all the data.
        """

        self.root_dir = root_dir
        self.categories = [
            "arccos",
            "arccosh",
            "arcsin",
            "arcsinh",
            "arctan",
            "arctan2",
            "arctanh",
            "heaviside",
            "log",
            "log10",
            "log1p",
            "log2",
            "multiply",
            "sin",
            "sinh",
            "sqrt",
            "square",
            "tan",
            "tanh",
            "divide",
            "add",
            "subtract",
            "linear",
            "exponential",
            "cubic",
            "joint_normal",
            "step",
            "quadratic",
            "w_shaped",
            "spiral",
            "logarithmic",
            "fourth_root",
            "sin_four_pi",
            "sin_sixteen_pi",
            "two_parabolas",
            "circle",
            "ellipse",
            "diamond",
            "multiplicative_noise",
            "multimodal_independence",
        ]

    def __len__(self):
        """return the total number of files in the directory."""
        return len(glob.glob(str(Path(self.root_dir, "*.pkl"))))

    def __getitem__(self, idx, img_size=128):
        """ get the item corresponding to the index and pad the samples
        to have img_size in both dimensions. Returns a tensor with all
        indices concatenated"""

        if torch.is_tensor(idx):
            idx = idx.tolist()

        final_maps_tensor = torch.empty(size=(0, 1, img_size, img_size))
        final_labels_tensor = torch.empty(size=(0, len(self.categories)))

        with open(Path(self.root_dir, str(idx) + ".pkl"), "rb") as file_path:
            data_dictionary = pickle.load(file_path)

        temp_map_tensor = torch.empty(
            size=(len(data_dictionary), 1, img_size, img_size)
        )
        temp_labels_tensor = torch.empty(
            size=(len(data_dictionary), len(self.categories))
        )

        for i, (_, entry) in enumerate(data_dictionary.items()):
            mgc_map, _, labels = entry

            # pad the datasets with wrong size
            if any(np.array(mgc_map.shape) != img_size):
                num_missing = np.ones(2) * img_size - np.array(mgc_map.shape)

                # if not even add one pad in that dimension,
                # and then progress as if the number of
                #  missing was even
                even = np.array(num_missing % 2 == 0)
                if not all(even):
                    # get the numerical values
                    odd = (~even).astype(int)
                    padding_mask = ((0, odd[0]), (0, odd[1]))
                    # pad 1 time
                    mgc_map = np.pad(mgc_map, pad_width=padding_mask, mode="edge")
                    # update the number of missing
                    num_missing = num_missing - even

                num_missing = (num_missing / 2).astype(int)
                padding_mask = ((num_missing[0],), (num_missing[1],))
                mgc_map = np.pad(mgc_map, pad_width=padding_mask, mode="edge")

            # set the temporary tensors
            temp_map_tensor[i, 0, :, :] = (
                torch.from_numpy(mgc_map).double().unsqueeze(0)
            )
            # get the numerical labels
            # this works for BCELoss
            temp_labels_tensor[i, :] = torch.from_numpy(
                np.isin(self.categories, labels)
            ).double()

        final_maps_tensor = torch.cat((final_maps_tensor, temp_map_tensor), dim=0)
        final_labels_tensor = torch.cat(
            (final_labels_tensor, temp_labels_tensor), dim=0
        )

        return final_maps_tensor, final_labels_tensor


class MinibatchDataset(Dataset):
    """Get a set of tensors and target classes
    a batch and return a minibatch for that batch"""

    def __init__(self, data):
        super().__init__()

        self.data = data

    def __len__(self):
        """return the total number of samples in the batch."""
        return self.data[1].shape[0]

    def __getitem__(self, idx):
        """ get the correct indices"""
        inputs, targets = self.data[0][idx, :, :, :], self.data[1][idx, :]
        return inputs, targets
        