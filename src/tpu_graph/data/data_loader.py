import os
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

import numba as nb
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
from tpu_graph import logger
from tpu_graph.constants import LOG_FEATURES, MAX_OP_CODE, DIM_FEATURES
from tpu_graph.utils.random_walk_pe import AddRandomWalkPE
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
        cache=False,
        cutoff: int = 16,
        clear_cache: bool = True,
        num_shards: int = 1,
        shard_id: int = 0,
        n_configs_per_file: int | None = None,
        prune: bool = False,
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
        :param num_shards: The number of shards to use
        :param shard_id: The shard to use
        :param n_configs_per_file: The number of configs per file to use
        :param prune: If True, the dataset is pruned such that only the configurable nodes and their inputs are used
        """

        # save the attributes
        self.cutoff = cutoff
        self.list_size = list_size
        self.list_shuffle = list_shuffle
        self.cache = cache
        self.clear_cache = clear_cache
        self.num_shards = num_shards
        self.shard_id = shard_id
        self.n_configs_per_file = n_configs_per_file
        self.prune = prune

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

                # for the indices (which will be used to combine thethe lists)
                if self.n_configs_per_file is not None:
                    size = min(len(data["config_runtime"]), self.n_configs_per_file)
                    indices = np.arange(size)
                else:
                    indices = np.arange(len(data["config_runtime"]))
                if self.list_shuffle:
                    np.random.shuffle(indices)
                self.index_dict[f] = indices

                # read out all the data if we want to cache
                if self.cache:
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
        if self.cache:
            data = self.data_dict[fname]
        else:
            with np.load(fname) as data:
                data = self.read_data(data)

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

    def _prune_data(self, data: dict[str : np.ndarray]):
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
                _data = np.load(cache_path, allow_pickle=True, mmap_mode="r")
                # read out the data
                _data_dict = {}
                _data_dict["node_feat"] = _data["node_feat"][:]
                _data_dict["node_opcode"] = _data["node_opcode"][:]
                _data_dict["edge_index"] = _data["edge_index"][:]
                _data_dict["pe"] = _data["pe"][:]
                _data_dict["new_pe"] = _data["new_pe_long"][:]
                _data_dict["node_feat_input"] = _data["node_feat_input"][:]
                _data_dict["node_config_ids"] = _data["node_config_ids"][:]

                # we only get a subset of the config features and runtimes
                if self.n_configs_per_file is not None:
                    indices = np.arange(len(_data["config_runtime"]))
                    np.random.shuffle(indices)
                    indices = indices[: self.n_configs_per_file]
                    _data_dict["config_runtime"] = _data["config_runtime"][indices]
                    try:
                        _data_dict["node_config_feat"] = self.read_config_memmap(cache_path, indices)
                    except Exception as e:
                        logger.error(f"Could not memmap {cache_path} because of {e}, using original file")
                        _data_dict["node_config_feat"] = _data["node_config_feat"][indices]
                else:
                    _data_dict["node_config_feat"] = _data["node_config_feat"][:]
                    _data_dict["config_runtime"] = _data["config_runtime"][:]

                # prune the data if necessary
                if self.prune:
                    _data_dict = self._prune_data(_data_dict)

                return _data_dict
            except Exception as e:
                logger.error(f"Could not load {cache_path} because of {e}")
                logger.error("Loading from original file")

        # we need to convert the data to a dict because of the caching
        data = {k: v for k, v in data.items()}

        # we add an additional node to the graph that is the output node
        outputs = np.where(data["node_feat"][:, 0] == 1)[0]
        new_edges = np.zeros((len(outputs), 2), dtype=np.int32)
        new_edges[:, 1] = outputs
        new_edges[:, 0] = len(data["node_feat"])
        new_edges = np.concatenate([data["edge_index"], new_edges], axis=0)

        # we flip the edges because of the different definition of the edge index (we copy to avoid negative strides)
        data["edge_index"] = np.fliplr(new_edges).T.copy()

        # we add an artificial node feature node and opcode
        data["node_feat"] = np.concatenate([data["node_feat"], np.zeros((1, data["node_feat"].shape[1]))], axis=0)
        data["node_opcode"] = np.concatenate([data["node_opcode"], np.array([MAX_OP_CODE - 1])], axis=0)

        # get the number of nodes
        n_nodes = data["node_feat"].shape[0]

        # the positional encoding
        data["pe"] = self.get_positional_encoding(n_nodes, data["edge_index"])

        # we write the file uncompressed back if caching is enabled
        # FIXME: Save again once the cache actually creates all fields
        # if self.cache:
        #     np.savez(cache_path, **data, allow_pickle=True)

        # we cut the data if we only want a subset of the configs
        if self.n_configs_per_file is not None:
            indices = data["indices"][: self.n_configs_per_file]
            data["node_config_feat"] = data["node_config_feat"][indices]
            data["config_runtime"] = data["config_runtime"][indices]

        # prune the data if necessary
        if self.prune:
            data = self._prune_data(data)

        return data

    def load_new_configs(self):
        """
        Loads new configs into the data dicts
        """

        if self.n_configs_per_file is None:
            logger.error("Cannot load new configs if n_configs_per_file is None")
            return

        # cycle through all files
        for fname in self.file_list:
            # get the cache name
            cache_file = self._fname_to_cache_path(fname)

            # load the data
            if self.cache and cache_file.exists():
                with np.load(cache_file, allow_pickle=True) as data:
                    indices = np.arange(len(data["config_runtime"]))
                    np.random.shuffle(indices)
                    indices = indices[: self.n_configs_per_file]
                    self.data_dict[fname]["config_runtime"] = data["config_runtime"][indices]

                    try:
                        # read out the data from the config
                        self.data_dict[fname]["node_config_feat"] = self.read_config_memmap(cache_file, indices)
                    except Exception as e:
                        logger.error(f"Could not memmap {cache_file} because of {e}, using original file")
                        self.data_dict[fname]["node_config_feat"] = data["node_config_feat"][indices]

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

    def __getitem__(self, idx):
        """
        Loads a file into memory and returns a sample
        :param idx: The index of the sample to return
        :return: The sample
        """

        # get data and indices
        data, indices = self.get_data_and_indices(idx)

        # read out the data for this graph (we copy because a subset will be logged)
        node_feat = data["node_feat"].copy()
        node_opcode = data["node_opcode"]
        pe = data["pe"]
        new_pe = data["new_pe"]
        edge_index = data["edge_index"]

        # we do everythin mod 128 (the TPU register length)
        dim_features = np.concatenate(
            [np.mod(node_feat[:, DIM_FEATURES] + 127, 128) / 128.0, np.floor(node_feat[:, DIM_FEATURES] / 128) / 10.0],
            axis=1,
        )
        # log some of the features (this can include dim features)
        node_feat[:, LOG_FEATURES] = np.log(node_feat[:, LOG_FEATURES] + 1)

        # add node_feat and pe
        node_feat = np.concatenate([node_feat, dim_features, pe, new_pe], axis=1)

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


class LayoutDatasetV2(LayoutDataset):
    """
    This dataset throws away all nodes which are not configurable
    """

    @staticmethod
    def _adapt_configs(configs: np.ndarray):
        """
        All configs with the default layout are adapted to get the -2 label
        :param configs: The configs to adapt (3D array)
        :return: The adapted configs
        """

        list_shape, n_nodes, _ = configs.shape
        new_config = configs.reshape(list_shape, n_nodes, 3, 6)
        mask = (new_config == -1).all(axis=-1)
        new_config[mask] = -2

        return new_config.reshape(list_shape, n_nodes, -1)

    def __getitem__(self, idx):
        """
        Loads a file into memory and returns a sample
        :param idx: The index of the sample to return
        :return: The sample
        """

        # get data and indices
        data, indices = self.get_data_and_indices(idx)

        # read out the data for this graph (we copy because a subset will be logged)
        node_feat = data["node_feat"].copy()
        # the input features are loged + 3 because of padding values of -2
        node_feat_input = data["node_feat_input"].copy()
        node_opcode = data["node_opcode"]
        pe = data["pe"]
        new_pe = data["new_pe"]
        edge_index = data["edge_index"]

        # mask the new features where there is no conv or dot
        node_feat_input[(node_opcode != 26) & (node_opcode != 34)] = -2.0
        node_feat_input = np.log(node_feat_input + 3.0)

        # we do everythin mod 128 (the TPU register length)
        dim_features = np.concatenate(
            [np.mod(node_feat[:, DIM_FEATURES] + 127, 128) / 128.0, np.floor(node_feat[:, DIM_FEATURES] / 128) / 10.0],
            axis=1,
        )
        # log some of the features (this can include dim features)
        node_feat = np.log(node_feat + 5) - np.log(5)

        # add node_feat and pe
        node_feat = np.concatenate([node_feat, node_feat_input, dim_features, pe, new_pe], axis=1)

        # get the configs and adapt them
        config_feat = data["node_config_feat"][indices]
        config_feat = self._adapt_configs(config_feat)

        # we divide by 5 to normalize the config features
        config_feat = config_feat / 5.0

        # the node config ids
        node_config_ids = data["node_config_ids"]

        # select only the configurable nodes
        node_feat = node_feat[node_config_ids]
        node_opcode = node_opcode[node_config_ids]

        # we normalize the runtime and multiply with the first to get quasi normalized time in nanoseconds
        config_runtime = data["config_runtime"][indices]
        node_opcode = np.tile(node_opcode[:, None], (self.list_size, 1, 1))
        # now where the config is default we add 128 to the opcode
        node_opcode[config_feat.sum(axis=2, keepdims=True) <= -17.5] += 128

        # tile the rest
        node_feat = np.tile(node_feat, (self.list_size, 1, 1))
        features = np.concatenate([node_opcode, node_feat, config_feat], axis=2)

        return features, edge_index, config_runtime


class LayoutDatasetV3(LayoutDataset):
    """
    Same as V2 but no adapt and pruning
    """

    def __getitem__(self, idx):
        """
        Loads a file into memory and returns a sample
        :param idx: The index of the sample to return
        :return: The sample
        """

        # get data and indices
        data, indices = self.get_data_and_indices(idx)

        # read out the data for this graph (we copy because a subset will be logged)
        node_feat = data["node_feat"].copy()
        node_opcode = data["node_opcode"]
        pe = data["pe"]
        new_pe = data["new_pe"]
        edge_index = data["edge_index"]
        node_feat_input = data["node_feat_input"].copy()

        # mask the new features where there is no conv or dot
        node_feat_input[(node_opcode != 26) & (node_opcode != 34)] = -2.0
        node_feat_input = np.log(node_feat_input + 3.0)

        # we do everythin mod 128 (the TPU register length)
        dim_features = np.concatenate(
            [
                np.mod(node_feat[:, DIM_FEATURES] + 127, 128) / 128.0,
                np.floor(node_feat[:, DIM_FEATURES] / 128) / 10.0,
            ],
            axis=1,
        )
        # log some of the features (this can include dim features)
        node_feat[:, LOG_FEATURES] = np.log(node_feat[:, LOG_FEATURES] + 1)

        # add node_feat and pe
        node_feat = np.concatenate([node_feat, node_feat_input, dim_features, pe, new_pe], axis=1)

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


class LayoutDatasetV4(LayoutDatasetV3):
    def _prune_data(self, data: dict[str : np.ndarray]):
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


class LayoutDatasetFinal(LayoutDataset):
    def __getitem__(self, idx):
        """
        Loads a file into memory and returns a sample
        :param idx: The index of the sample to return
        :return: The sample
        """

        # get data and indices
        data, indices = self.get_data_and_indices(idx)

        # read out the data for this graph (we copy because a subset will be logged)
        node_feat = data["node_feat"].copy()
        node_opcode = data["node_opcode"]
        pe = data["pe"]
        new_pe = data["new_pe"]
        edge_index = data["edge_index"]
        node_feat_input = data["node_feat_input"].copy()

        # mask the new features where there is no conv or dot
        node_feat_input[(node_opcode != 26) & (node_opcode != 34)] = -2.0
        node_feat_input = np.log(node_feat_input + 3.0)

        # we do everythin mod 128 (the TPU register length)
        dim_features = np.concatenate(
            [np.mod(node_feat[:, DIM_FEATURES] + 127, 128) / 128.0, np.floor(node_feat[:, DIM_FEATURES] / 128) / 10.0],
            axis=1,
        )

        # log the node features
        node_feat = np.log(node_feat + 5) - np.log(5)

        # add node_feat and pe
        node_feat = np.concatenate([node_feat, node_feat_input, dim_features, pe, new_pe], axis=1)

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

        # the selection indices
        selection_indices = np.array([np.arange(len(node_config_ids)), node_config_ids])

        return features, edge_index, selection_indices, config_runtime

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
        edge_offset = 0
        select_indices = []
        select_offset = 0
        # the individual length of the graphs
        edge_lengths = []
        index_lengths = []

        # unpack everything
        for t in tensors:
            assert len(t) == 4, "The length of the tensors must be 4"
            node_feat, edge_index, select_index, config_runtime = t

            # append the tensors that need to go through the network
            features.append(node_feat)
            times.append(config_runtime)
            edge_indices.append(edge_index + edge_offset)
            edge_offset += node_feat.shape[1]
            edge_lengths.append(node_feat.shape[1])
            select_indices.append(select_index + select_offset)
            select_offset += select_index.shape[1]
            index_lengths.append(select_index.shape[1])

        # stack the tensors
        features = torch.Tensor(np.concatenate(features, axis=1))
        times = torch.tensor(np.stack(times, axis=0), dtype=dtype)

        # the connection matrix
        edge_indices = torch.tensor(np.concatenate(edge_indices, axis=1), dtype=torch.long)
        select_indices = torch.tensor(np.concatenate(select_indices, axis=1), dtype=torch.long)

        return features, [edge_lengths, index_lengths], times, torch.concatenate([edge_indices, select_indices], dim=1)
