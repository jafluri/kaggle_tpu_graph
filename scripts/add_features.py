import os
from pathlib import Path

import click
import numpy as np
from tpu_graph import logger
from tpu_graph.proto import tuning_pb2
from tpu_graph.utils.feature_extraction import get_additional_features
from tpu_graph.utils.random_walk_pe import compute_pe_rwpe, compute_pe_tg
from tpu_graph.constants import LOG_FEATURES, DIM_FEATURES, MAX_OP_CODE


@click.command()
@click.option(
    "--proto_dir",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    help="Directory containing the protobuf files, will be searched recursively",
)
@click.option(
    "--npz_dir",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    help="Directory containing the npz files, will be searched recursively",
)
@click.option("--padding", type=int, default=-1, help="The padding value to use for the new features")
@click.option(
    "--pe_dim_asym",
    type=int,
    default=16,
    help="The number of positional encoding vectors to use for the asymmetric positional encoding",
)
@click.option(
    "--pe_dim_sym",
    type=int,
    default=112,
    help="The number of positional encoding vectors to use for the symmetric positional encoding",
)
def add_features(
    proto_dir: list[str | bytes | os.PathLike],
    npz_dir: list[str | bytes | os.PathLike],
    padding: int,
    pe_dim_asym: int,
    pe_dim_sym: int,
):
    # list all files
    proto_files = sorted(list(Path(proto_dir).glob("**/*.pb")))
    npz_files = [f for f in sorted(list(Path(npz_dir).glob("**/*.npz"))) if not str(f).endswith("cached.npz")]
    assert len(proto_files) == len(npz_files), f"Found {len(proto_files)} proto files and {len(npz_files)} npz files"
    logger.info(f"Found {len(proto_files)} files")

    # make sure they are all identical
    for proto_file, npz_file in zip(proto_files, npz_files):
        if proto_file.stem != npz_file.stem:
            raise ValueError(f"Files {proto_file} and {npz_file} do not match")

    # cycle through everything
    for proto_file, npz_file in zip(proto_files, npz_files):
        logger.info(f"Processing {proto_file}")

        # init the module
        t_data = tuning_pb2.ModuleTuningData()
        with open(proto_file, "rb") as f:
            t_data.ParseFromString(f.read())
        m = t_data.module

        # load the file
        npz_data = dict(np.load(npz_file))
        n_nodes = len(npz_data["node_opcode"])
        logger.info(f"Loaded {n_nodes} nodes")

        # we add an additional node to the graph that is the output node
        outputs = np.where(npz_data["node_feat"][:, 0] == 1)[0]
        new_edges = np.zeros((len(outputs), 2), dtype=np.int32)
        new_edges[:, 1] = outputs
        new_edges[:, 0] = len(npz_data["node_feat"])
        new_edges = np.concatenate([npz_data["edge_index"], new_edges], axis=0)

        # we flip the edges because of the different definition of the edge index (we copy to avoid negative strides)
        npz_data["edge_index"] = np.fliplr(new_edges).T.copy()

        # we add an artificial node feature node and opcode
        npz_data["node_feat"] = np.concatenate(
            [npz_data["node_feat"], np.zeros((1, npz_data["node_feat"].shape[1]))], axis=0
        )
        npz_data["node_opcode"] = np.concatenate([npz_data["node_opcode"], np.array([MAX_OP_CODE - 1])], axis=0)

        # get the new features
        logger.info("Extracting new features from protobuf")
        input_features, new_features = get_additional_features(m, npz_data, padding=padding)

        # log some of the features with large ranges (shift to make them positive)
        node_feat = npz_data["node_feat"]
        node_feat[:, LOG_FEATURES] = np.log(node_feat[:, LOG_FEATURES] + 1)
        new_features[:, [4, 5, 6, 7]] = np.log(new_features[:, [4, 5, 6, 7]] + 2)
        input_features = np.log(input_features + 3)

        # create the dim features
        dim_features = np.concatenate(
            [
                np.mod(node_feat[:, DIM_FEATURES] + 127, 128) / 128.0,
                np.floor(node_feat[:, DIM_FEATURES] / 128) / 10.0,
            ],
            axis=1,
        )

        # add the new features
        npz_data["node_feat"] = np.concatenate([node_feat, new_features, input_features, dim_features], axis=1)

        # get the pe
        logger.info("Computing positional encodings")

        pe_asym = compute_pe_tg(
            edge_index=npz_data["edge_index"], n_nodes=n_nodes, num_lpe_vecs=pe_dim_asym, device="cuda"
        )
        pe_sym = compute_pe_rwpe(npz_data["edge_index"], n_nodes=n_nodes, num_lpe_vecs=pe_dim_sym, device="cuda")

        # add the pe
        npz_data["pe_asym"] = pe_asym
        npz_data["pe_sym"] = pe_sym

        # save the file with cached.npz
        logger.info("Saving file")
        cache_name = npz_file.parent.joinpath(npz_file.stem + "_cached.npz")
        np.savez(cache_name, **npz_data)


if __name__ == "__main__":
    add_features()
