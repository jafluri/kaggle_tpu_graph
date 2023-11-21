import os
from collections import defaultdict
from pathlib import Path
from typing import Literal
from zipfile import ZipFile

import numba as nb
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
from tpu_graph import logger
from tpu_graph.constants import MAX_OP_CODE
from tqdm import tqdm


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


def load_from_npz(zf, name):
    """
    Loads a array from an npz file into a memmap
    :param zf: The open npz file (open as ZipFile)
    :param name: The name of the array to read out
    :return: The memmap
    """
    # figure out offset of .npy in .npz
    info = zf.NameToInfo[name + ".npy"]
    assert info.compress_type == 0
    # don't ask me why it is 20 bytes, but it works
    try:
        zf.fp.seek(info.header_offset + len(info.FileHeader()) + 20)
        # read .npy header
        version = np.lib.format.read_magic(zf.fp)
        np.lib.format._check_version(version)
    except Exception:
        zf.fp.seek(info.header_offset + len(info.FileHeader()) - 20)
        # read .npy header
        version = np.lib.format.read_magic(zf.fp)
        np.lib.format._check_version(version)
    shape, fortran_order, dtype = np.lib.format._read_array_header(zf.fp, version)
    offset = zf.fp.tell()
    # create memmap
    return np.memmap(
        zf.filename, dtype=dtype, shape=shape, order="F" if fortran_order else "C", mode="r", offset=offset
    )


class LayoutDataset(Dataset):
    """
    This is a base class for all datasets that are used in the TPUGraph project
    """

    def __init__(
        self,
        data_path: str | bytes | os.PathLike | list[str | bytes | os.PathLike],
        list_size: int = 32,
        list_shuffle: bool = False,
        num_shards: int = 1,
        shard_id: int = 0,
        n_configs_per_file: int | None = None,
        prune: None | Literal["v1", "v2", "v3"] = None,
        log_features: bool = True,
    ):
        """
        Inits the dataset with a directory containing the NPZ files
        :param data_path: The directory containing the NPZ files that will be loaded, can also be a list of directories
        :param list_size: The number of samples that are returned per graph
        :param list_shuffle: If True, the samples are shuffled before returning them
        :param num_shards: The number of shards to use
        :param shard_id: The shard to use
        :param n_configs_per_file: The number of configs per file to use
        :param prune: Prune the graph. There are the following options:
                        None: No pruning
                        v1: Prune all nodes besides the configurable ones. The edges fully remvoed
                        v2: Prune all nodes besides the configurable ones and their inputs/outputs
                        v3: Prune all nodes besides the configurable ones, their inputs/outputs and merge the rest ïnto
                            virtual nodes
        :param log_features: If True, the features are logged
        """

        # save the attributes
        self.list_size = list_size
        self.list_shuffle = list_shuffle
        self.num_shards = num_shards
        self.shard_id = shard_id
        self.n_configs_per_file = n_configs_per_file
        self.prune = prune
        self.log = log_features

        # get all the files
        if not isinstance(data_path, list):
            data_path = [data_path]
        self.data_path = sorted([Path(p) for p in data_path])

        # get all the files
        self.file_list = []
        for path in self.data_path:
            file_list = sorted(path.glob("*_cached.npz"))
            logger.info(f"Found {len(file_list)} files in {path}")
            self.file_list.extend(file_list)

        # now we split the files into shards
        self.file_list = self.file_list[shard_id::num_shards]
        logger.info(f"Using {len(self.file_list)} files for shard {shard_id}")

        # we need open all files once to get the size of the dataset
        logger.info("Loading all files to get the size of the dataset")
        # list for the sizes of the data, the data and the indices
        self.size_list = []
        self.data_dict = {}
        self.index_dict = {}
        for f in tqdm(self.file_list):
            with np.load(f) as data:
                # get the size of the dataset, note that we might not load all configs
                size = len(data["config_runtime"]) // self.list_size
                if self.n_configs_per_file is not None:
                    size = min(size, self.n_configs_per_file // self.list_size)
                self.size_list.append(size)

                # for the indices (which will be used to combine the lists)
                if self.n_configs_per_file is not None:
                    size = min(len(data["config_runtime"]), self.n_configs_per_file)
                    indices = np.arange(size)
                else:
                    indices = np.arange(len(data["config_runtime"]))
                if self.list_shuffle:
                    np.random.shuffle(indices)
                self.index_dict[f] = indices

                # read out the data
                self.data_dict[f] = self.read_data(data)

        self.length = sum(self.size_list)
        self.offsets = np.cumsum(self.size_list)
        logger.info(f"The dataset has a total size of {self.length}")

    def reshuffle_indices(self):
        """
        Reshuffles the indices of the dataset to create new possible list sets
        """

        for v in self.index_dict.values():
            np.random.shuffle(v)

    @staticmethod
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
        features = torch.Tensor(np.concatenate(features, axis=1))
        times = torch.tensor(np.stack(times, axis=0), dtype=dtype)

        # the connection matrix
        edge_indices = torch.tensor(np.concatenate(edge_indices, axis=1), dtype=torch.long)

        return features, lengths, times, edge_indices

    def get_dataloader(
        self,
        batch_size: int,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = True,
        num_workers: int = 2,
    ):
        """
        Returns a dataloader for the dataset
        :param batch_size: The batch size to use
        :param shuffle: If True, the dataset is shuffled
        :param pin_memory: If True, the memory is pinned
        :param drop_last: If True, the last batch is dropped if it is not full
        :param num_workers: The number of workers to use
        :return: The dataloader
        """

        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=pin_memory,
            collate_fn=self.collate_fn_tiles,
            drop_last=drop_last,
            num_workers=num_workers,
        )

    def get_data_and_indices(self, idx):
        """
        Given an index to fetch a sample, returns the data of the right file and the indices to read out
        :param idx: Index to fetch the sample
        :return: The data and the file name
        """

        # get the file
        file_idx = np.searchsorted(self.offsets, idx, side="right")

        # get the offset
        if file_idx == 0:
            offset = idx
        else:
            offset = idx - self.offsets[file_idx - 1]

        # get the indices
        fname = self.file_list[file_idx]
        indices = self.index_dict[fname][offset * self.list_size : (offset + 1) * self.list_size]

        # load the file
        data = self.data_dict[fname]

        return data, indices

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

    def read_config_memmap(self, fname, indices):
        """
        Reads config features from memmap
        :param fname: The npz file
        :param indices: The indices to read out
        :return: The data
        """

        # here we need to do some black magic. NPZ files are loaded into normal arrays and not memmaps
        zf = ZipFile(fname)
        node_config_feat = load_from_npz(zf, "node_config_feat")[indices]

        # clean up
        zf.close()

        return node_config_feat

    def _prune_data_v1(self, data: dict[str : np.ndarray]):
        """
        Prunes the data such that only the configurable nodes are used
        :param data: The data dict
        :return: The same data dict with the pruned data
        """

        # readout
        node_feat = data["node_feat"]
        node_feat_input = data["node_feat_input"]
        node_opcode = data["node_opcode"]
        pe = data["pe"]
        new_pe = data["new_pe"]

        # read the config ids
        node_config_ids = data["node_config_ids"]

        # remove all egdes
        edge_index = np.zeros((2, 0), dtype=np.int64)

        # filter the data
        node_feat = node_feat[node_config_ids]
        node_feat_input = node_feat_input[node_config_ids]
        node_opcode = node_opcode[node_config_ids]
        pe = pe[node_config_ids]
        new_pe = new_pe[node_config_ids]

        # the new config ids are just the indices
        node_config_ids = np.arange(len(node_config_ids))

        # update the data
        data["node_feat"] = node_feat
        data["node_feat_input"] = node_feat_input
        data["node_opcode"] = node_opcode
        data["pe"] = pe
        data["new_pe"] = new_pe
        data["edge_index"] = edge_index
        data["node_config_ids"] = node_config_ids

        return data

    def _prune_data_v2(self, data: dict[str : np.ndarray]):
        """
        Prunes the data such that only the configurable nodes and their inputs are used
        :param data: The data dict
        :return: The same data dict with the pruned data
        """

        # get the edges and the node ids
        edge_index = data["edge_index"]
        node_ids = data["node_config_ids"]

        # get the number of nodes
        n_nodes = len(data["node_opcode"])
        mask = np.zeros(n_nodes, dtype=bool)

        # get the configurable nodes
        mask[node_ids] = True

        # we add all edges that lead to a configurable node or are the output of a configurable node
        new_edges = edge_index[:, mask[edge_index[1]] | mask[edge_index[0]]]

        # add everything to mask
        mask[new_edges[0]] = True
        mask[new_edges[1]] = True

        # the edges still refer to the old node ids, we need to map them to the new ones
        id_map = np.arange(n_nodes)
        id_map[mask] = np.arange(mask.sum())
        new_edges[0] = id_map[new_edges[0]]
        new_edges[1] = id_map[new_edges[1]]

        # prune the data
        data["edge_index"] = new_edges
        data["node_feat"] = data["node_feat"][mask]
        data["node_opcode"] = data["node_opcode"][mask]
        data["node_feat_input"] = data["node_feat_input"][mask]
        data["pe"] = data["pe"][mask]
        data["new_pe"] = data["new_pe"][mask]

        # new config ids are the indices of the config ids on the pruned mask
        data["node_config_ids"] = id_map[node_ids]

        return data

    def _prune_data_v3(self, data: dict[str : np.ndarray]):
        """
        Prunes the data such that only the configurable nodes and their inputs are used and merges the rest of the nodes
        into virtual nodes
        :param data: The data dict
        :return: The same data dict with the pruned data
        """

        # read out the data
        node_config_ids = data["node_config_ids"]
        edge_index = data["edge_index"]

        # get a set with all important node
        node_set = set(node_config_ids)

        # set that contains all configs ids and its input and output nodes
        extended_node_set = set(node_config_ids)
        for i, o in edge_index.T:
            if i in node_config_ids or o in node_config_ids:
                extended_node_set.add(i)
                extended_node_set.add(o)

        # create a dict where each node has its input and outputs
        graph_dict = defaultdict(lambda: defaultdict(set))

        # cycle through all edges
        for i, o in edge_index.T:
            graph_dict[o]["inputs"].add(i)
            graph_dict[i]["outputs"].add(o)

        # function to merge two nodes
        def merge_nodes(graph_dict, i, j):
            """
            Merges two nodes i, j -> i of a given graph_dict
            """

            # redirect all nodes that have j as output to i
            for node_in in graph_dict[j]["inputs"]:
                graph_dict[node_in]["outputs"].remove(j)
                if node_in != i:
                    graph_dict[node_in]["outputs"].add(i)
                    graph_dict[i]["inputs"].add(node_in)

            # all outputs of j are now outputs of i
            for node_out in graph_dict[j]["outputs"]:
                graph_dict[node_out]["inputs"].remove(j)
                if node_out != i:
                    graph_dict[node_out]["inputs"].add(i)
                    graph_dict[i]["outputs"].add(node_out)

            del graph_dict[j]

        # merge all nodes that are not in the node set
        merged = []
        while True:
            # get the current start node
            current_node = None
            for k in graph_dict.keys():
                if k not in extended_node_set and k not in merged:
                    current_node = k
            if current_node is None:
                break

            # merge all nodes connected to the start node
            while True:
                merge_list = []
                for i in graph_dict[current_node]["inputs"]:
                    if i not in extended_node_set:
                        merge_list.append(i)
                if len(merge_list) == 0:
                    break

                for o in graph_dict[current_node]["outputs"]:
                    if o not in extended_node_set:
                        merge_list.append(o)
                for m in merge_list:
                    merge_nodes(graph_dict, current_node, m)

            # mark as merged
            merged.append(current_node)

        # get all dummy nodes
        dummies = []
        for k in graph_dict.keys():
            if k not in node_set:
                dummies.append(k)
        num_dummies = len(dummies)

        # we create a map old_index -> new_index
        index_map = {}
        old_ids = dummies + list(node_config_ids)
        for new_id, old_id in enumerate(old_ids):
            index_map[old_id] = new_id

        # create the new edges
        new_edges = []
        for k, v in graph_dict.items():
            for i in v["inputs"]:
                new_edges.append([index_map[i], index_map[k]])
            for o in v["outputs"]:
                new_edges.append([index_map[k], index_map[o]])
        new_edges = np.array(new_edges).T

        # set the indices
        data["edge_index"] = new_edges

        # features, set dummies to 0
        feat = data["node_feat"][old_ids]
        feat[:num_dummies] = 0.0

        # opcode, set dummies to MAX_OP_CODE
        opcode = data["node_opcode"][old_ids]
        opcode[:num_dummies] = MAX_OP_CODE - 1

        # pe, set dummies to 0
        pe = data["pe"][old_ids]
        pe[:num_dummies] = 0.0

        # new_pe, set dummies to 0
        new_pe = data["new_pe"][old_ids]
        new_pe[:num_dummies] = 0.0

        # node_feat_input, set dummies to -2
        feat_input = data["node_feat_input"][old_ids]
        feat_input[:num_dummies] = -2.0

        # node_config_ids
        node_config_ids = np.arange(len(node_config_ids)) + num_dummies

        # set the data
        data["node_feat"] = feat
        data["node_opcode"] = opcode
        data["pe"] = pe
        data["new_pe"] = new_pe
        data["node_feat_input"] = feat_input
        data["node_config_ids"] = node_config_ids

        return data

    def read_data(self, data: dict[str : np.ndarray]):
        """
        Reads out the datadict from the npz file into memory and adds the imaginary output node and creates the graph
        :param data: The data of the npz file (not necessary in memory)
        :return: The data dict with the imaginary output node and the graph
        """

        # get the file name
        filename = data.zip.filename

        # read out the data
        _data_dict = {}
        _data_dict["node_feat"] = data["node_feat"][:]
        _data_dict["node_opcode"] = data["node_opcode"][:]
        _data_dict["edge_index"] = data["edge_index"][:]
        _data_dict["pe_asym"] = data["pe_asym"][:]
        _data_dict["pe_sym"] = data["pe_sym"][:]
        _data_dict["node_config_ids"] = data["node_config_ids"][:]

        # we only get a subset of the config features and runtimes
        if self.n_configs_per_file is not None:
            indices = np.arange(len(data["config_runtime"]))
            np.random.shuffle(indices)
            indices = indices[: self.n_configs_per_file]
            _data_dict["config_runtime"] = data["config_runtime"][indices]
            try:
                _data_dict["node_config_feat"] = self.read_config_memmap(filename, indices)
            except Exception as e:
                logger.error(f"Could not memmap {filename} because of {e}, using original file")
                _data_dict["node_config_feat"] = data["node_config_feat"][indices]
        else:
            _data_dict["node_config_feat"] = data["node_config_feat"][:]
            _data_dict["config_runtime"] = data["config_runtime"][:]

        # prune the data if necessary
        if self.prune == "v1":
            _data_dict = self._prune_data_v1(_data_dict)
        elif self.prune == "v2":
            _data_dict = self._prune_data_v2(_data_dict)
        elif self.prune == "v3":
            _data_dict = self._prune_data_v3(_data_dict)
        else:
            raise ValueError(f"Unknown prune option {self.prune}")

        return _data_dict

    def load_new_configs(self):
        """
        Loads new configs into the data dicts
        """

        if self.n_configs_per_file is None:
            logger.error("Cannot load new configs if n_configs_per_file is None")
            return

        # cycle through all files
        for fname in self.file_list:
            with np.load(fname, allow_pickle=True) as data:
                indices = np.arange(len(data["config_runtime"]))
                np.random.shuffle(indices)
                indices = indices[: self.n_configs_per_file]
                self.data_dict[fname]["config_runtime"] = data["config_runtime"][indices]

                try:
                    # read out the data from the config
                    self.data_dict[fname]["node_config_feat"] = self.read_config_memmap(fname, indices)
                except Exception as e:
                    logger.error(f"Could not memmap {fname} because of {e}, using original file")
                    self.data_dict[fname]["node_config_feat"] = data["node_config_feat"][indices]

    def __len__(self):
        """
        Returns the length of the dataset
        :return: The length of the dataset
        """

        return self.length

    def __getitem__(self, idx):
        """
        Loads a file into memory and returns a sample
        :param idx: The index of the sample to return
        :return: The sample
        """

        # get data and indices
        data, indices = self.get_data_and_indices(idx)

        # read out the data for this graph (we copy because a subset will be logged)
        node_feat = data["node_feat"]
        node_opcode = data["node_opcode"]
        pe = data["pe_asym"]
        new_pe = data["pe_sym"]
        edge_index = data["edge_index"]

        # add node_feat and pe
        if self.log:
            node_feat = np.log(node_feat + 5.0) - np.log(5.0)
        node_feat = np.concatenate([node_feat, pe, new_pe], axis=1)

        # we divide by 5 to normalize the config features
        config_feat = data["node_config_feat"][indices] / 5.0
        node_config_ids = data["node_config_ids"]

        # fill the config features
        config_feat = fill_config_feat(len(node_feat), node_config_ids, config_feat)

        # get the runtime
        config_runtime = data["config_runtime"][indices]

        # tile config_features such that axis 0 matches with the number of nodes
        node_opcode = np.tile(node_opcode[:, None], (self.list_size, 1, 1))
        node_feat = np.tile(node_feat, (self.list_size, 1, 1))
        features = np.concatenate([node_opcode, node_feat, config_feat], axis=2)

        return features, edge_index, config_runtime
