import os
from abc import ABCMeta, abstractmethod
from pathlib import Path

from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
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

    def __init__(
        self,
        data_path: str | bytes | os.PathLike | list[str | bytes | os.PathLike],
        list_size: int = 32,
        list_shuffle: bool = False,
        cache=False,
        cutoff: int = 3,
        lpe_dim: int = 16,
        clear_cache: bool = False,
    ):
        """
        Inits the dataset with a directory containing the NPZ files
        :param data_path: The directory containing the NPZ files that will be loaded, can also be a list of directories
        :param list_size: The number of samples that are returned per graph
        :param list_shuffle: If True, the samples are shuffled before returning them
        :param cache: If True, the dataset is cached in memory and the preprocessed data is saved
        :param cutoff: The cutoff for the neighborhood in the connection matrix
        :param lpe_dim: The dimension of the laplacian positional encoding
        :param clear_cache: If True, the cache is cleared, meaning the preprocessed data is ignored
        """

        # save the attributes
        self.cutoff = cutoff
        self.list_size = list_size
        self.list_shuffle = list_shuffle
        self.lpe_dim = lpe_dim
        self.clear_cache = clear_cache

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
        row_indices = []
        col_indices = []
        offset = 0
        # the individual length of the graphs
        lengths = []

        # unpack everything
        for t in tensors:
            assert len(t) == 3, "The length of the tensors must be 3"
            node_feat, connection_matrix, config_runtime = t
            row_ids, col_ids = connection_matrix

            # append the tensors that need to go through the network
            features.append(node_feat)
            lengths.append(node_feat.shape[1])
            times.append(config_runtime)
            row_indices.append(row_ids + offset)
            col_indices.append(col_ids + offset)
            offset += node_feat.shape[1]

        # stack the tensors
        features = torch.Tensor(np.concatenate(features, axis=1)).to(device)
        times = torch.tensor(np.stack(times, axis=0), dtype=dtype).to(device)

        # the connection matrix
        row_indices = torch.tensor(np.concatenate(row_indices, axis=0), dtype=torch.long).to(device)
        col_indices = torch.tensor(np.concatenate(col_indices, axis=0), dtype=torch.long).to(device)
        connection_matrix = (row_indices, col_indices)

        return features, lengths, times, connection_matrix

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

    def add_imaginary_output_node(self, node_feat: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
        """
        Adds an imaginary output node to the node features and edge index and converts the layout to iGraph
        :param node_feat: The node features, used to find the output nodes
        :param edge_index: The array containing the edge indices
        :return: The new edges
        """

        # we add an imaginary node that connects all the output nodes
        outputs = np.where(node_feat[:, 0] == 1)[0]
        new_edges = np.zeros((len(outputs), 2), dtype=np.int32)
        new_edges[:, 1] = outputs
        new_edges[:, 0] = len(node_feat)
        new_edges = np.concatenate([edge_index, new_edges], axis=0)

        # we flip the edges because of the different definition of the edge index (we copy to avoid negative strides)
        new_edges = np.fliplr(new_edges).copy()

        return new_edges

    def get_graph_and_context_matrix(self, n_nodes: int, edge_index: np.ndarray):
        """
        Returns the graph and the connection matrix for the given node features and edge index
        :param n_nodes: The number of nodes in the graph
        :param edge_index: The edge index
        :return: The graph and the sparse context matrix (mixed)
        """

        # create the graph, name it after the file name such that it has a unique name
        graph = Graph(n=n_nodes, edges=edge_index, directed=True)

        # get the connection matrix (COO format) without the imaginary node
        row_ids = []
        col_ids = []
        for vertex_id, neighborhood in enumerate(graph.neighborhood(order=self.cutoff, mode="in", mindist=0)[:-1]):
            row_ids.extend([vertex_id] * len(neighborhood))
            col_ids.extend(neighborhood)

        # create the sparse connection matrix
        connection_mat = (np.array(row_ids, dtype=np.int32), np.array(col_ids, dtype=np.int32))

        return graph, connection_mat

    def calculate_lpe(self, graph: Graph):
        """
        Calculates the laplacian positional encoding for the given graph, taken from
        pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/transforms/add_positional_encoding.html
        :param graph: The graph to calculate the lpe for
        :return: The lpe for the graph
        """

        # get the laplacian matrix
        graph = graph.as_undirected()
        laplacian = np.array(graph.laplacian(normalized=True))

        # get the eigenvectors
        try:
            eig_vals, eig_vecs = eigsh(
                laplacian,
                k=self.lpe_dim + 1 if self.lpe_dim < laplacian.shape[0] else laplacian.shape[0] - 1,
                which="SA",
                return_eigenvectors=True,
            )
        except Exception as e:
            logger.error(f"Could not calculate the eigenvectors for {graph} because of {e}")
            logger.error("Using eigh instead")
            eig_vals, eig_vecs = eigh(laplacian)

        # sort the eigenvectors
        eig_vecs = np.real(eig_vecs[:, eig_vals.argsort()])
        eig_vecs = eig_vecs[:, 1 : self.lpe_dim + 1]

        # pad if necessary
        if eig_vecs.shape[1] != self.lpe_dim:
            eig_vecs = np.pad(eig_vecs, ((0, 0), (0, self.lpe_dim - eig_vecs.shape[1])))

        return eig_vecs

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

        # read out the data for this graph
        node_feat = data["node_feat"]
        data["edge_index"] = self.add_imaginary_output_node(node_feat, data["edge_index"])

        # get graph and connection matrix
        graph, connection_matrix = self.get_graph_and_context_matrix(
            n_nodes=len(node_feat) + 1, edge_index=data["edge_index"]
        )
        graph["name"] = filename

        # unpack the connection matrix and save in the data dict
        data["row_indices"], data["col_indices"] = connection_matrix

        # calculate the lpe
        data["lpe"] = self.calculate_lpe(graph)

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
        lpe = data["lpe"]
        connection_matrix = (data["row_indices"], data["col_indices"])

        # read out the specific config
        indices = data["indices"][offset * self.list_size : (offset + 1) * self.list_size]
        config_feat = data["config_feat"][indices]

        # we normalize the runtime and multiply with the first to get quasi normalized time in nanoseconds
        config_runtime = data["config_runtime"][indices] / data["config_runtime_normalizers"][indices]

        # tile config_features such that axis 0 matches with the number of nodes
        config_feat = np.tile(config_feat, (node_feat.shape[0], 1, 1)).transpose((1, 0, 2))
        node_feat = np.tile(node_feat, (self.list_size, 1, 1))
        lpe = np.tile(lpe[:-1], (self.list_size, 1, 1))
        node_opcode = np.tile(node_opcode[:, None], (self.list_size, 1, 1))
        features = np.concatenate([node_opcode, node_feat[:], lpe, config_feat], axis=2)

        return features, connection_matrix, config_runtime


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
        lpe = data["lpe"]
        connection_matrix = (data["row_indices"], data["col_indices"])

        # read out the specific config
        indices = data["indices"][offset * self.list_size : (offset + 1) * self.list_size]
        config_feat = data["node_config_feat"][indices]
        node_config_ids = data["node_config_ids"]

        # fill the config features
        config_feat = fill_config_feat(len(node_feat), node_config_ids, config_feat)

        # we normalize the runtime and multiply with the first to get quasi normalized time in nanoseconds
        config_runtime = data["config_runtime"][indices]

        # tile config_features such that axis 0 matches with the number of nodes
        lpe = np.tile(lpe[:-1], (self.list_size, 1, 1))
        node_opcode = np.tile(node_opcode[:, None], (self.list_size, 1, 1))
        node_feat = np.tile(node_feat, (self.list_size, 1, 1))
        features = np.concatenate([node_opcode, node_feat, lpe, config_feat], axis=2)

        return features, connection_matrix, config_runtime
