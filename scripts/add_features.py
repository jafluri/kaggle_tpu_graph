import os
from pathlib import Path

import click
import numpy as np
from tpu_graph import logger
from tpu_graph.proto import tuning_pb2
from tpu_graph.utils.feature_extraction import get_additional_features
from tpu_graph.utils.random_walk_pe import compute_pe


@click.command()
@click.option("--proto_dir", type=click.Path(exists=True, dir_okay=True, file_okay=False))
@click.option("--npz_dir", type=click.Path(exists=True, dir_okay=True, file_okay=False))
@click.option("--padding", type=int, default=-1)
@click.option("--pe_dim_asym", type=int, default=16)
@click.option("--pe_dim_sym", type=int, default=112)
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

        # we flip the edges because of the different definition of the edge index (we copy to avoid negative strides)
        npz_data["edge_index"] = np.fliplr(npz_data["edge_index"]).T.copy()

        # get the new features
        logger.info("Extracting new features from protobuf")
        input_features, new_features = get_additional_features(m, npz_data, padding=padding)

        # add the new features
        npz_data["node_feat"] = np.concatenate([npz_data["node_feat"], input_features, new_features], axis=1)

        # get the pe
        logger.info("Computing positional encodings")

        pe_asym = compute_pe(
            edge_index=npz_data["edge_index"], n_nodes=n_nodes, num_lpe_vecs=pe_dim_asym, device="cuda"
        )
        pe_sym = compute_pe(
            npz_data["edge_index"], n_nodes=n_nodes, num_lpe_vecs=pe_dim_sym, symmetric=True, device="cuda"
        )

        # add the pe
        npz_data["pe_asym"] = pe_asym
        npz_data["pe_sym"] = pe_sym

        # save the file with cached.npz
        logger.info("Saving file")
        cache_name = npz_file.parent.joinpath(npz_file.stem + "_cached.npz")
        np.savez_compressed(cache_name, **npz_data)


if __name__ == "__main__":
    add_features()
