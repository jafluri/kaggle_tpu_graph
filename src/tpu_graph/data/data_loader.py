import os
from abc import ABCMeta, abstractmethod
from pathlib import Path

import numba as nb
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
from tpu_graph import logger
from tpu_graph.constants import LOG_FEATURES
from tpu_graph.utils.random_walk_pe import AddRandomWalkPE
from tqdm import tqdm


class TPUGraphDataset(Dataset, metaclass=ABCMeta):
    """
    This is a base class for all datasets that are used in the TPUGraph project
    """

    def __init__(
        self,
        data_path: str | bytes | os.PathLike | list[str | bytes | os.PathLike],
        list_size: int = 32,
        list_shuffle: bool = False,
        cache=False,
        cutoff: int = 16,
        clear_cache: bool = True,
    ):
        """
        Inits the dataset with a directory containing the NPZ files
        :param data_path: The directory containing the NPZ files that will be loaded, can also be a list of directories
        :param list_size: The number of samples that are returned per graph
        :param list_shuffle: If True, the samples are shuffled before returning them
        :param cache: If True, the dataset is cached in memory and the preprocessed data is saved
        :param cutoff: The cutoff for the neighborhood in the connection matrix
        :param decay: The decay for the neighborhood in the connection matrix
        :param clear_cache: If True, the cache is cleared, meaning the preprocessed data is ignored
        """

        # save the attributes
        self.cutoff = cutoff
        self.list_size = list_size
        self.list_shuffle = list_shuffle
        self.clear_cache = clear_cache

        # create the encoder
        self.encoder = AddRandomWalkPE(
            self.cutoff,
        )

        # get all the files
        if not isinstance(data_path, list):
            data_path = [data_path]
        self.data_path = sorted([Path(p) for p in data_path])

        # get all the files
        self.file_list = []
        for path in self.data_path:
            file_list = [f for f in sorted(path.glob("*.npz")) if not f.name.endswith("_cached.npz")]
            logger.info(f"Found {len(file_list)} files in {path}")
            self.file_list.extend(file_list)

        # we need open all files once to get the size of the dataset
        logger.info("Loading all files to get the size of the dataset")
        self.size_list = []
        self.cache = cache
        self.data_dict = {}
        for f in tqdm(self.file_list):
            with np.load(f) as data:
                self.size_list.append(len(data["config_runtime"]) // self.list_size)
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
        # for the connection matrices
        edge_indices = []
        offset = 0
        # the individual length of the graphs
        lengths = []

        # unpack everything
        for t in tensors:
            assert len(t) == 3, "The length of the tensors must be 3"
            node_feat, edge_index, config_runtime = t

            # append the tensors that need to go through the network
            features.append(node_feat)
            lengths.append(node_feat.shape[1])
            times.append(config_runtime)
            edge_indices.append(edge_index + offset)
            offset += node_feat.shape[1]

        # stack the tensors
        features = torch.Tensor(np.concatenate(features, axis=1)).to(device)
        times = torch.tensor(np.stack(times, axis=0), dtype=dtype).to(device)

        # the connection matrix
        edge_indices = torch.tensor(np.concatenate(edge_indices, axis=1), dtype=torch.long).to(device)

        return features, lengths, times, edge_indices

    def get_dataloader(self, batch_size: int, shuffle: bool = True, pin_memory: bool = False, drop_last: bool = True):
        """
        Returns a dataloader for the dataset
        :param batch_size: The batch size to use
        :param shuffle: If True, the dataset is shuffled
        :param pin_memory: If True, the memory is pinned
        :param drop_last: If True, the last batch is dropped if it is not full
        :return: The dataloader
        """

        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=pin_memory,
            collate_fn=self.collate_fn_tiles,
            drop_last=drop_last,
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

    def get_positional_encoding(self, n_nodes, edge_index):
        """
        Returns the positional encoding for the given graph
        :param n_nodes: The number of nodes in the graph
        :param edge_index: The edge index
        :return: The positional encoding
        """

        with torch.no_grad():
            # create the graph (we need to add self loops and reverse edges)
            # edge_index = np.concatenate([edge_index, np.flipud(edge_index)], axis=1)
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            edge_index = add_self_loops(edge_index, num_nodes=n_nodes)[0]
            x = torch.ones((n_nodes, 1))
            data = Data(x=x, edge_index=edge_index)

            # get the positional encoding
            data = self.encoder(data)

        return data.random_walk_pe.cpu().numpy()

    def read_data(self, data: dict[str : np.ndarray]):
        """
        Reads out the datadict from the npz file into memory and adds the imaginary output node and creates the graph
        :param data: The data of the npz file (not necessary in memory)
        :return: The data dict with the imaginary output node and the graph
        """

        # read out the data
        filename = data.zip.filename
        cache_path = self._fname_to_cache_path(filename)

        if self.clear_cache and cache_path.exists():
            os.remove(cache_path)

        # if we cached the file we load it from the cache
        if self.cache and cache_path.exists():
            try:
                _data = np.load(cache_path, allow_pickle=True)
                _data = {k: v for k, v in _data.items()}
                return _data
            except Exception as e:
                logger.error(f"Could not load {cache_path} because of {e}")
                logger.error("Loading from original file")

        # we need to convert the data to a dict because of the caching
        data = {k: v for k, v in data.items()}

        # we flip the edges because of the different definition of the edge index (we copy to avoid negative strides)
        data["edge_index"] = np.fliplr(data["edge_index"]).T.copy()

        # get the number of nodes
        n_nodes = data["node_feat"].shape[0]

        # the positional encoding
        data["pe"] = self.get_positional_encoding(n_nodes, data["edge_index"])

        # the indices of the samples
        data["indices"] = np.arange(len(data["config_runtime"]))
        if self.list_shuffle:
            np.random.shuffle(data["indices"])

        # we write the file uncompressed back if caching is enabled
        if self.cache:
            np.savez(cache_path, **data, allow_pickle=True)

        return data

    def _fname_to_cache_path(self, fname):
        """
        Returns the path to the cached file
        :param fname: The filename
        :return: The path to the cached file
        """

        fname = Path(fname)
        cache_name = fname.parent.joinpath(fname.stem + "_cached.npz")

        return cache_name

    def __len__(self):
        """
        Returns the length of the dataset
        :return: The length of the dataset
        """

        return self.length

    @abstractmethod
    def __getitem__(self, idx):
        """
        This has to be implemented by the child classed. Returns a sample from the dataset.
        """


@nb.njit(nb.float32[:, :, :](nb.int64, nb.int64[:], nb.float32[:, :, :]))
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
    padded_config_feat = np.zeros((node_config_feat.shape[0], n_nodes, node_config_feat.shape[2]), dtype=np.float32)
    for num, idx in enumerate(node_config_ids):
        padded_config_feat[:, idx, :] = node_config_feat[:, num, :]

    return padded_config_feat


class TileDataset(TPUGraphDataset):
    """
    This class implements the dataset for the tiles. It loads all the files and provides an interface to them.
    """

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
        pe = data["pe"]
        edge_index = data["edge_index"]

        # add node_feat and pe
        node_feat = np.concatenate([node_feat, pe], axis=1)

        # read out the specific config
        indices = data["indices"][offset * self.list_size : (offset + 1) * self.list_size]
        config_feat = data["config_feat"][indices]

        # we normalize the runtime and multiply with the first to get quasi normalized time in nanoseconds
        config_runtime = data["config_runtime"][indices] / data["config_runtime_normalizers"][indices]

        # tile config_features such that axis 0 matches with the number of nodes
        config_feat = np.tile(config_feat, (node_feat.shape[0], 1, 1)).transpose((1, 0, 2))
        node_feat = np.tile(node_feat, (self.list_size, 1, 1))
        node_opcode = np.tile(node_opcode[:, None], (self.list_size, 1, 1))
        features = np.concatenate([node_opcode, node_feat[:], config_feat], axis=2)

        return features, edge_index, config_runtime


class LayoutDataset(TPUGraphDataset):
    """
    This class implements the dataset for the layout. It loads all the files and provides an interface to them.
    """

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
        pe = data["pe"]
        edge_index = data["edge_index"]

        # log some of the features
        node_feat[:, LOG_FEATURES] = np.log(node_feat[:, LOG_FEATURES] + 1)

        # add node_feat and pe
        node_feat = np.concatenate([node_feat, pe], axis=1)

        # read out the specific config
        indices = data["indices"][offset * self.list_size : (offset + 1) * self.list_size]

        # we divide by 5 to normalize the config features
        config_feat = data["node_config_feat"][indices] / 5.0
        node_config_ids = data["node_config_ids"]

        # fill the config features
        config_feat = fill_config_feat(len(node_feat), node_config_ids, config_feat)

        # we normalize the runtime and multiply with the first to get quasi normalized time in nanoseconds
        config_runtime = data["config_runtime"][indices]

        # tile config_features such that axis 0 matches with the number of nodes
        node_opcode = np.tile(node_opcode[:, None], (self.list_size, 1, 1))
        node_feat = np.tile(node_feat, (self.list_size, 1, 1))
        features = np.concatenate([node_opcode, node_feat, config_feat], axis=2)

        return features, edge_index, config_runtime
