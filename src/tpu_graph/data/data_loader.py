import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from tpu_graph import logger
from tqdm import tqdm


class TileDataset(Dataset):
    """
    This class implements the dataset for the tiles. It loads all the files and provides an interface to them.
    """

    def __init__(self, data_path: str | bytes | os.PathLike):
        """
        Inits the dataset with a directory containing the NPZ files
        :param data_path: The directory containing the NPZ files used for the training of the tiles network
        """

        # get all the files
        self.data_path = Path(data_path)
        self.file_list = sorted(self.data_path.glob("*.npz"))
        logger.info(f"Fount {len(self.file_list)} files in {self.data_path}")

        # we need open all files once to get the size of the dataset
        logger.info("Loading all files to get the size of the dataset")
        self.size_list = []
        for f in tqdm(self.file_list):
            with np.load(f) as data:
                self.size_list.append(len(data["config_runtime"]))

        self.length = sum(self.size_list)
        self.offsets = np.cumsum(self.size_list)
        logger.info(f"The dataset has a total size of {self.length}")

    def __getitem__(self, idx):
        """
        Loads a file into memory and returns a sample
        :param idx: The index of the sample to return
        :return: The sample
        """

        # get the file
        file_idx = np.searchsorted(self.offsets, idx, side="right")

        # get the offset
        if file_idx == 0:
            offset = idx
        else:
            offset = idx - self.offsets[file_idx - 1]

        # load the file
        with np.load(self.file_list[file_idx]) as data:
            # read out the data for this graph
            node_feat = data["node_feat"]
            node_opcode = data["node_opcode"]
            edge_index = data["edge_index"]

            # we add an imaginary node that connects all the output nodes
            outputs = np.where(node_feat[:, 0] == 1)[0]
            new_edges = np.zeros((len(outputs), 2), dtype=np.int32)
            new_edges[:, 1] = outputs
            new_edges[:, 0] = len(node_feat)
            edge_index = np.concatenate([edge_index, new_edges], axis=0)

            # read out the specific config
            config_feat = data["config_feat"][offset]

            # we normalize the runtime and multiply with the first to get quasi normalized time in nanoseconds
            config_runtime = data["config_runtime"][offset] / data["config_runtime_normalizers"][offset]
            config_runtime *= data["config_runtime"][0]

        # tile config_features such that axis 0 matches with the number of nodes
        config_feat = np.tile(config_feat, (node_feat.shape[0], 1))
        features = np.concatenate([node_feat, node_opcode[:, None], config_feat], axis=1)
        return features, config_runtime, edge_index

    def __len__(self):
        """
        Returns the length of the dataset
        :return: The length of the dataset
        """

        return self.length


def collate_fn_tiles(tensors: list[tuple], dtype=torch.float32):
    """
    A custom collate function for the tiles dataset
    :param tensors: A list of tuples that are returned by the dataset
    :param dtype: The dtype to use for the tensors
    :return: The collated output for the dataloader
    """

    # list for the collection
    features = []
    times = []
    edge_indices = []

    # unpack everything
    for t in tensors:
        assert len(t) == 3, "The length of the tensors must be 3"
        node_feat, config_runtime, edge_index = t

        # append the tensors that need to go through the network
        features.append(torch.tensor(node_feat, dtype=dtype))
        times.append(torch.tensor(config_runtime, dtype=dtype))
        # the edge index needs to be int32
        edge_indices.append(torch.tensor(edge_index, dtype=torch.int32))

    return features, times, edge_indices
