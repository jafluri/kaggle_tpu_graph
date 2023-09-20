import os
from abc import ABCMeta, abstractmethod
from pathlib import Path

import numba as nb
import numpy as np
import torch
from igraph import Graph
from torch.utils.data import Dataset
from tpu_graph import logger
from tqdm import tqdm


class TPUGraphDataset(Dataset, metaclass=ABCMeta):
    """
    This is a base class for all datasets that are used in the TPUGraph project
    """

    def __init__(self, data_path: str | bytes | os.PathLike, cache=False):
        """
        Inits the dataset with a directory containing the NPZ files
        :param data_path: The directory containing the NPZ files used for the training of the tiles network
        :param cache: If True, the dataset is cached in memory
        """

        # get all the files
        self.data_path = Path(data_path)
        self.file_list = sorted(self.data_path.glob("*.npz"))
        logger.info(f"Found {len(self.file_list)} files in {self.data_path}")

        # we need open all files once to get the size of the dataset
        logger.info("Loading all files to get the size of the dataset")
        self.size_list = []
        self.cache = cache
        self.data_dict = {}
        for f in tqdm(self.file_list):
            with np.load(f) as data:
                self.size_list.append(len(data["config_runtime"]))
                # read out all the data if we want to cache
                if self.cache:
                    # read out the data
                    self.data_dict[f] = self.read_data(data)

        self.length = sum(self.size_list)
        self.offsets = np.cumsum(self.size_list)
        logger.info(f"The dataset has a total size of {self.length}")

    @staticmethod
    def collate_fn_tiles(tensors: list[tuple], dtype=torch.float32, device="cuda"):
        """
        A custom collate function for the tiles dataset
        :param tensors: A list of tuples that are returned by the dataset
        :param dtype: The dtype to use for the tensors
        :param device: The device to put the tensors on, note the edges and graphs are always on the CPU
        :return: The collated output for the dataloader
        """

        # list for the collection
        features = []
        times = []
        edge_indices = []
        graphs = []

        # unpack everything
        for t in tensors:
            assert len(t) == 4, "The length of the tensors must be 4"
            node_feat, config_runtime, edge_index, graph = t

            # append the tensors that need to go through the network
            features.append(torch.tensor(node_feat, dtype=dtype).to(device))
            times.append(torch.tensor(config_runtime, dtype=dtype).to(device))
            # the edge index needs to be int32
            edge_indices.append(torch.tensor(edge_index, dtype=torch.int32))
            graphs.append(graph)

        return features, times, edge_indices, graphs

    def get_dataloader(self, batch_size: int, shuffle: bool = True, num_workers: int = 0, pin_memory: bool = False):
        """
        Returns a dataloader for the dataset
        :param batch_size: The batch size to use
        :param shuffle: If True, the dataset is shuffled
        :param num_workers: The number of workers to use for the dataloader
        :param pin_memory: If True, the memory is pinned
        :return: The dataloader
        """

        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=self.collate_fn_tiles,
        )

    def get_data_and_offset(self, idx):
        """
        Given an index to fetch a sample, returns the data of the right file and the offset (index of the sample in the
        file)
        :param idx: Index to fetch the sample
        :return: The data and the offset
        """

        # get the file
        file_idx = np.searchsorted(self.offsets, idx, side="right")

        # get the offset
        if file_idx == 0:
            offset = idx
        else:
            offset = idx - self.offsets[file_idx - 1]

        # load the file
        if self.cache:
            data = self.data_dict[self.file_list[file_idx]]
        else:
            with np.load(self.file_list[file_idx]) as data:
                data = self.read_data(data)

        return data, offset

    def __len__(self):
        """
        Returns the length of the dataset
        :return: The length of the dataset
        """

        return self.length

    @staticmethod
    @abstractmethod
    def read_data(data: dict[str : np.ndarray]):
        """
        This has to be implemented by the child classes. Reads out the datadict from the npz file into memory and adds
        """
        pass

    @abstractmethod
    def __getitem__(self, idx):
        """
        This has to be implemented by the child classed. Returns a sample from the dataset.
        """


@nb.njit(nb.float32[:, :](nb.int64, nb.int64[:], nb.float32[:, :]))
def fill_config_feat(n_nodes, node_config_ids: np.ndarray, node_config_feat: np.ndarray):
    """
    This routine is meant for the layout dataset.
    Creates a full config feature vector with shape (n_nodes, n_config_features) from the sparse config feature vector
    :param n_nodes: The number of nodes in the whole graph
    :param node_config_ids: The ID of the config vectors that map the node_config_feature vectors to node ids
    :param node_config_feat: The config features of the configurable nodes
    :return: The fully padded config feature vectors
    """

    # we make fill all missing config features with 0
    padded_config_feat = np.zeros((n_nodes, node_config_feat.shape[1]), dtype=np.float32)
    for num, idx in enumerate(node_config_ids):
        padded_config_feat[idx, :] = node_config_feat[num, :]

    return padded_config_feat


class TileDataset(TPUGraphDataset):
    """
    This class implements the dataset for the tiles. It loads all the files and provides an interface to them.
    """

    @staticmethod
    def read_data(data: dict[str : np.ndarray]):
        """
        Reads out the datadict from the npz file into memory and adds the imaginary output node and creates the graph
        :param data: The data of the npz file (not necessary in memory)
        :return: The data dict with the imaginary output node and the graph
        """

        # read out the data
        filename = data.zip.filename
        data = {k: v for k, v in data.items()}

        # read out the data for this graph
        node_feat = data["node_feat"]
        edge_index = data["edge_index"]

        # we add an imaginary node that connects all the output nodes
        outputs = np.where(node_feat[:, 0] == 1)[0]
        new_edges = np.zeros((len(outputs), 2), dtype=np.int32)
        new_edges[:, 1] = outputs
        new_edges[:, 0] = len(node_feat)
        new_edges = np.concatenate([edge_index, new_edges], axis=0)

        # we flip the edges because of the different definition of the edge index (we copy to avoid negative strides)
        new_edges = np.fliplr(new_edges).copy()
        data["edge_index"] = new_edges

        # create the graph, name it after the file name such that it has a unique name
        graph = Graph(n=len(node_feat) + 1, edges=new_edges, directed=True)
        graph["name"] = filename
        data["graph"] = graph

        return data

    def __getitem__(self, idx):
        """
        Loads a file into memory and returns a sample
        :param idx: The index of the sample to return
        :return: The sample
        """

        # get data and offset
        data, offset = self.get_data_and_offset(idx)

        # read out the data for this graph
        node_feat = data["node_feat"]
        node_opcode = data["node_opcode"]
        edge_index = data["edge_index"]
        graph = data["graph"]

        # read out the specific config
        config_feat = data["config_feat"][offset]

        # we normalize the runtime and multiply with the first to get quasi normalized time in nanoseconds
        config_runtime = data["config_runtime"][offset] / data["config_runtime_normalizers"][offset]
        config_runtime *= data["config_runtime"][0]

        # tile config_features such that axis 0 matches with the number of nodes
        config_feat = np.tile(config_feat, (node_feat.shape[0], 1))
        features = np.concatenate([node_feat, node_opcode[:, None], config_feat], axis=1)

        return features, config_runtime, edge_index, graph


class LayoutDataset(TPUGraphDataset):
    """
    This class implements the dataset for the layout. It loads all the files and provides an interface to them.
    """

    @staticmethod
    def read_data(data: dict[str : np.ndarray]):
        """
        Reads out the datadict from the npz file into memory and adds the imaginary output node and creates the graph
        :param data: The data of the npz file (not necessary in memory)
        :return: The data dict with the imaginary output node and the graph
        """

        # read out the data
        filename = data.zip.filename
        data = {k: v for k, v in data.items()}

        # read out the data for this graph
        node_feat = data["node_feat"]
        edge_index = data["edge_index"]

        # we add an imaginary node that connects all the output nodes
        outputs = np.where(node_feat[:, 0] == 1)[0]
        new_edges = np.zeros((len(outputs), 2), dtype=np.int32)
        new_edges[:, 1] = outputs
        new_edges[:, 0] = len(node_feat)
        new_edges = np.concatenate([edge_index, new_edges], axis=0)

        # we flip the edges because of the different definition of the edge index (we copy to avoid negative strides)
        new_edges = np.fliplr(new_edges).copy()
        data["edge_index"] = new_edges

        # create the graph, name it after the file name such that it has a unique name
        graph = Graph(n=len(node_feat) + 1, edges=new_edges, directed=True)
        graph["name"] = filename
        data["graph"] = graph

        return data

    def __getitem__(self, idx):
        """
        Loads a file into memory and returns a sample
        :param idx: The index of the sample to return
        :return: The sample
        """

        # get data and offset
        data, offset = self.get_data_and_offset(idx)

        # read out the data for this graph
        node_feat = data["node_feat"]
        node_opcode = data["node_opcode"]
        edge_index = data["edge_index"]
        graph = data["graph"]

        # read out the specific config
        config_feat = data["node_config_feat"][offset]
        node_config_ids = data["node_config_ids"]

        # fill the config features
        config_feat = fill_config_feat(len(node_feat), node_config_ids, config_feat)

        # we normalize the runtime and multiply with the first to get quasi normalized time in nanoseconds
        config_runtime = data["config_runtime"][offset]

        # tile config_features such that axis 0 matches with the number of nodes
        features = np.concatenate([node_feat, node_opcode[:, None], config_feat], axis=1)

        return features, config_runtime, edge_index, graph
