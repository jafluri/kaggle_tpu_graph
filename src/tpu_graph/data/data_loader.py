import os
from abc import ABCMeta, abstractmethod
from pathlib import Path

from scipy.sparse.linalg import eigsh
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
        list_size: int = 16,
        list_shuffle: bool = False,
        cache=False,
        cutoff: int = 3,
        decay: float = 0.5,
        lpe_dim: int = 16,
    ):
        """
        Inits the dataset with a directory containing the NPZ files
        :param data_path: The directory containing the NPZ files that will be loaded, can also be a list of directories
        :param list_size: The number of samples that are returned per graph
        :param list_shuffle: If True, the samples are shuffled before returning them
        :param cache: If True, the dataset is cached in memory
        :param cutoff: The cutoff for the shortest path connection matrix
        :param decay: The decay for the shortest path connection matrix
        :param lpe_dim: The dimension of the laplacian positional encoding
        """

        # save the attributes
        self.cutoff = cutoff
        self.list_size = list_size
        self.list_shuffle = list_shuffle
        self.decay = decay
        self.lpe_dim = lpe_dim

        # get all the files
        if not isinstance(data_path, list):
            data_path = [data_path]
        self.data_path = sorted([Path(p) for p in data_path])

        # get all the files
        self.file_list = []
        for path in self.data_path:
            file_list = sorted(path.glob("*.npz"))[:5]
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
        indices = []
        values = []
        offset = 0
        # the individual length of the graphs
        lengths = []

        # unpack everything
        for t in tensors:
            assert len(t) == 3, "The length of the tensors must be 3"
            node_feat, connection_matrix, config_runtime = t

            # append the tensors that need to go through the network
            features.append(node_feat)
            lengths.append(node_feat.shape[1])
            times.append(config_runtime)
            indices.append(connection_matrix._indices() + offset)
            offset += node_feat.shape[1]
            values.append(connection_matrix._values())

        # stack the tensors
        features = torch.Tensor(np.concatenate(features, axis=1)).to(device)
        times = torch.tensor(np.stack(times, axis=0), dtype=dtype).to(device)
        indices = torch.cat(indices, dim=1)
        values = torch.cat(values, dim=0)
        connection_matrix = (
            torch.sparse_coo_tensor(indices, values, (features.shape[1], features.shape[1], values.shape[1]))
            .float()
            .to(device)
        )

        return features, lengths, times, connection_matrix

    def get_dataloader(self, batch_size: int, shuffle: bool = True, pin_memory: bool = False):
        """
        Returns a dataloader for the dataset
        :param batch_size: The batch size to use
        :param shuffle: If True, the dataset is shuffled
        :param pin_memory: If True, the memory is pinned
        :return: The dataloader
        """

        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
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
        vals = []
        for vertex_id, neighborhood in enumerate(graph.neighborhood(order=self.cutoff, mode="in", mindist=0)[:-1]):
            if len(neighborhood) == 1:
                lpe_subgraph = np.ones((1, self.lpe_dim))
            else:
                subgraph = graph.subgraph(neighborhood)
                lpe_subgraph = self.calculate_lpe(subgraph)
            for neighbor, lpe in zip(neighborhood, lpe_subgraph):
                row_ids.append(vertex_id)
                col_ids.append(neighbor)
                vals.append(lpe)

        # create the sparse connection matrix
        connection_mat = torch.sparse_coo_tensor(
            np.stack([row_ids, col_ids]), vals, (n_nodes - 1, n_nodes - 1, self.lpe_dim)
        )

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
        laplacian = np.array(graph.laplacian(mode="in", normalized=True))

        # get the eigenvectors
        eig_vals, eig_vecs = eigsh(
            laplacian,
            k=self.lpe_dim + 1 if self.lpe_dim < laplacian.shape[0] else laplacian.shape[0] - 1,
            which="SA",
            return_eigenvectors=True,
        )

        # sort the eigenvectors
        eig_vecs = np.real(eig_vecs[:, eig_vals.argsort()])

        # pad if necessary
        if self.lpe_dim < laplacian.shape[0]:
            eig_vecs = eig_vecs[:, 1:]
        else:
            eig_vecs = np.pad(eig_vecs, ((0, 0), (0, self.lpe_dim - laplacian.shape[0] + 1)))

        return eig_vecs

    def read_data(self, data: dict[str : np.ndarray]):
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
        data["edge_index"] = self.add_imaginary_output_node(node_feat, data["edge_index"])

        # get graph and connection matrix
        graph, connection_matrix = self.get_graph_and_context_matrix(
            n_nodes=len(node_feat) + 1, edge_index=data["edge_index"]
        )
        graph["name"] = filename
        data["graph"] = graph
        data["connection_matrix"] = connection_matrix

        # calculate the lpe
        data["lpe"] = self.calculate_lpe(graph)

        # the indices of the samples
        data["indices"] = np.arange(len(data["config_runtime"]))
        if self.list_shuffle:
            np.random.shuffle(data["indices"])

        return data

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
        connection_matrix = data["connection_matrix"]

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
        edge_index = data["edge_index"]
        graph = data["graph"]
        connection_matrix = data["connection_matrix"]

        # read out the specific config
        config_feat = data["node_config_feat"][offset]
        node_config_ids = data["node_config_ids"]

        # fill the config features
        config_feat = fill_config_feat(len(node_feat), node_config_ids, config_feat)

        # we normalize the runtime and multiply with the first to get quasi normalized time in nanoseconds
        config_runtime = data["config_runtime"][offset]

        # tile config_features such that axis 0 matches with the number of nodes
        features = np.concatenate([node_opcode[:, None], node_feat, config_feat], axis=1)

        return features, config_runtime, edge_index, connection_matrix, graph
