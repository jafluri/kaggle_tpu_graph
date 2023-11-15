import datetime
import logging
import os
import sys
from pathlib import Path

import click
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from tpu_graph.data import LayoutDatasetV4 as LayoutDataset
from tpu_graph.networks import TPUGraphNetwork
from tpu_graph.training import evaluation
from tpu_graph.training.ltr.pairwise_losses import PairwiseHingeLoss
from tqdm import tqdm

import wandb


def setup(rank, world_size):
    """
    Setup the distributed training
    :param rank: The rank of the current process
    :param world_size: The total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=36000))
    # dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=36000))


def cleanup():
    dist.destroy_process_group()


def train_network(rank, kwargs):
    # create a logger for the training
    logger = logging.getLogger(f"tpu_network.train.rank_{rank}")
    log_formatter = logging.Formatter(
        fmt="%(asctime)s %(name)10s %(levelname).3s   %(message)s ", datefmt="%y-%m-%d %H:%M:%S", style="%"
    )
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_formatter)
    logger.addHandler(stream_handler)
    logger.propagate = False
    logger.setLevel(logging.INFO)

    # setup the distributed training
    logger.info("Starting training of the tile network")
    setup(rank, kwargs["world_size"])

    # setup wandb
    if rank == 0:
        # Start with the wandb init
        logger.info("Starting wandb")
        wandb.init(
            mode="disabled",
            project="TPU Graph",
            config={
                "learning_rate": kwargs["learning_rate"],
                "dataset": "Tiles Dataset of the TPU Graph Benchmark",
                "epochs": kwargs["epochs"],
                "batch_size": kwargs["batch_size"],
                "cosine_annealing": kwargs["cosine_annealing"],
                "network_type": "Layout Network",
                "list_size": kwargs["list_size"],
                "dropout": kwargs["dropout"],
                "reload_configs": kwargs["reload_configs"],
                "steps_per_batch": kwargs["steps_per_batch"],
                "n_configs_per_file": kwargs["n_configs_per_file"],
                "n_configs_val": kwargs["n_configs_val"],
            },
        )

        # get the name
        run_name = wandb.run.name

        # print the run name
        logger.info(f"Run ID: {run_name}")
    else:
        run_name = "foo"

    # broadcast the run name
    object_list = [run_name]
    dist.broadcast_object_list(object_list, src=0, device=torch.device(rank))
    run_name = object_list[0]
    logger.info(f"Run ID: {run_name}")

    # load the dataset
    base_paths = [Path(p) for p in kwargs["data_path"]]

    logger.info("Loading the dataset for training")
    train_dataset = LayoutDataset(
        [base_path.joinpath("train") for base_path in base_paths],
        cache=kwargs["cache"],
        clear_cache=kwargs["clear_cache"],
        list_size=kwargs["list_size"],
        list_shuffle=True,
        num_shards=kwargs["world_size"],
        shard_id=rank,
        n_configs_per_file=kwargs["n_configs_per_file"],
        prune=True,
    )
    train_dataloader = train_dataset.get_dataloader(batch_size=kwargs["batch_size"])

    logger.info("Loading the dataset for validation")
    val_dataset = LayoutDataset(
        [base_path.joinpath("valid") for base_path in base_paths],
        cache=kwargs["cache"],
        list_size=1,
        clear_cache=kwargs["clear_cache"],
        num_shards=kwargs["world_size"],
        shard_id=rank,
        n_configs_per_file=kwargs["n_configs_val"],
        prune=True,
    )
    val_dataloader = val_dataset.get_dataloader(batch_size=32, shuffle=False, drop_last=False)

    logger.info("Loading the dataset for testing")
    test_dataset = LayoutDataset(
        [base_path.joinpath("test") for base_path in base_paths],
        cache=kwargs["cache"],
        list_size=1,
        clear_cache=kwargs["clear_cache"],
        num_shards=kwargs["world_size"],
        shard_id=rank,
        prune=True,
    )
    test_dataloader = test_dataset.get_dataloader(batch_size=16, shuffle=False, drop_last=False)

    # we build a super simple network for starters
    logger.info("Building the network")

    network = TPUGraphNetwork(
        embedding_out=512,
        message_network_dims=[256, 256, 256],
        n_normal_features=140 + 30 + 16,
        n_dim_features=2 * 37,
        n_lpe_features=128,
        n_configs=18,
        embedding_dim=128,
        lpe_embedding_dim=64,
        message_dim=128,
        linformer_dim=256,
        embedding_version="v2",
    )

    # network to GPU
    network = network.to(rank)
    network = DDP(network, device_ids=[rank])

    # restore the model if necessary
    if kwargs["restore_path"] is not None:
        logger.info("Restoring the model")
        # configure map_location properly
        map_location = {"cuda:%d" % 0: "cuda:%d" % rank}
        network.load_state_dict(torch.load(kwargs["restore_path"], map_location=map_location))

    # get the optimizer
    optimizer = optim.Adam(network.parameters(), lr=kwargs["learning_rate"], weight_decay=kwargs["weight_decay"])

    # get the total number of batches per epoch
    total = torch.tensor(len(train_dataloader)).to(rank)
    logger.info(f"Total number of batches per epoch (local): {len(train_dataloader)}")
    dist.all_reduce(total, op=dist.ReduceOp.MIN, async_op=False)
    total = total.item()
    logger.info(f"Total number of batches per epoch (global): {total}")
    if kwargs["max_train_steps"] is not None:
        total = np.minimum(kwargs["max_train_steps"], len(train_dataloader))

    # get the scheduler
    scheduler = None
    if kwargs["cosine_annealing"]:
        t_max = total * kwargs["epochs"]
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, t_max, eta_min=1e-6)

    # create the saving path
    save_path = Path(kwargs["save_path"])
    save_path.mkdir(parents=True, exist_ok=True)

    # create the loss fn
    loss_class = PairwiseHingeLoss()
    batch_pad = torch.ones(kwargs["batch_size"]).long().to(rank) * train_dataset.list_size

    def loss_fn(pred, label):
        return loss_class(pred, label, batch_pad)

    # start the training loop
    logger.info("Starting the training loop")
    for epoch in range(kwargs["epochs"]):
        network.train()
        if rank == 0:
            logger.info(f"Starting epoch {epoch}")
            pbar = tqdm(train_dataloader, postfix={"loss": 0}, total=total)
        else:
            pbar = train_dataloader
        for batch_idx, (features, lengths, runtimes, edge_index) in enumerate(pbar):
            # to GPU
            features = features.to(rank)
            runtimes = runtimes.to(rank)
            edge_index = edge_index.to(rank)

            # predict the runtimes
            for _ in range(kwargs["steps_per_batch"]):
                _, pred_runtimes = network(features, edge_index, lengths)
                loss = torch.mean(loss_fn(pred_runtimes, runtimes))

                # backprop
                optimizer.zero_grad()
                loss.backward()

                # clip the gradients and step
                torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
                optimizer.step()

            # step the scheduler
            if scheduler is not None:
                scheduler.step()

            # log the loss to the logger
            total_loss = loss.detach() / kwargs["world_size"]
            dist.reduce(total_loss, dst=0, op=dist.ReduceOp.SUM, async_op=False)
            if rank == 0:
                # set postfix
                pbar.set_postfix({"loss": total_loss.item()})

                # log the summaries
                wandb.log(
                    {
                        "loss": total_loss.item(),
                        "lr": scheduler.get_last_lr()[0],
                    }
                )

            # break if necessary
            if batch_idx >= total - 1:
                break

        # save the network for this epoch
        if rank == 0:
            logger.info("Saving the model")
            torch.save(network.state_dict(), save_path.joinpath(f"{run_name}_{epoch=}.pt"))

        logger.info("Validating the network")
        network.eval()
        avg_kendall = evaluation.evaluate_layout_network(
            network,
            val_dataloader,
            save_path.joinpath(f"{run_name}_{rank=}_{epoch=}_val.npz"),
            device=rank,
        )
        # log everything
        logger.info(f"Average kendall for epoch {epoch} (local): {avg_kendall}")
        total_avg_kendall = torch.tensor(avg_kendall).to(rank) / kwargs["world_size"]
        dist.all_reduce(total_avg_kendall, op=dist.ReduceOp.SUM, async_op=False)
        avg_kendall = total_avg_kendall.item()
        logger.info(f"Average kendall for epoch {epoch} (global): {avg_kendall}")
        if rank == 0:
            wandb.log({"val_avg_kendall": avg_kendall})

        # test the network
        logger.info("Testing the network")
        avg_kendall = evaluation.evaluate_layout_network(
            network,
            test_dataloader,
            save_path.joinpath(f"{run_name}_{rank=}_{epoch=}_test.npz"),
            device=rank,
        )
        # log everything
        logger.info(f"Average kendall for epoch {epoch} (local): {avg_kendall}")
        total_avg_kendall = torch.tensor(avg_kendall).to(rank) / kwargs["world_size"]
        dist.all_reduce(total_avg_kendall, op=dist.ReduceOp.SUM, async_op=False)
        avg_kendall = total_avg_kendall.item()
        logger.info(f"Average kendall for epoch {epoch} (global): {avg_kendall}")

        # reshuffle the dataset
        if kwargs["n_configs_per_file"] is not None and epoch < kwargs["epochs"] - 1:
            if (epoch + 1) % kwargs["reload_configs"] == 0:
                logger.info("Loading new configs...")
                train_dataset.load_new_configs()
            else:
                logger.info("Shuffling the dataset")
                train_dataset.reshuffle_indices()
            train_dataloader = train_dataset.get_dataloader(batch_size=kwargs["batch_size"])
        elif kwargs["n_configs_per_file"] is not None and epoch < kwargs["epochs"] - 1:
            logger.info("Shuffling the dataset")
            train_dataset.reshuffle_indices()
            train_dataloader = train_dataset.get_dataloader(batch_size=kwargs["batch_size"])

    # save the model
    if rank == 0:
        logger.info("Saving the final model")
        torch.save(network.state_dict(), save_path.joinpath(f"{run_name}.pt"))

    # cleanup
    cleanup()


@click.command()
@click.option(
    "--data_path",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    help="The path to the data, for multiple datasets, one can specify this argument multiple times."
    "Each directory should contain a train, valid and test directory.",
    multiple=True,
)
@click.option(
    "--save_path",
    default=Path("./models"),
    type=click.Path(dir_okay=True, file_okay=False),
    help="The path to save the model",
)
@click.option(
    "--restore_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False, file_okay=True),
    help="The path to restore the model",
)
@click.option("--learning_rate", type=float, default=0.001, help="The learning rate to use for training")
@click.option(
    "--cosine_annealing",
    is_flag=True,
    help="If set, the learning rate is annealed using the cosine annealing scheduler",
)
@click.option("--epochs", type=int, default=1, help="The number of epochs to train")
@click.option("--batch_size", type=int, default=16, help="The batch size to use for training")
@click.option("--cache", is_flag=True, help="If set, the dataset is cached in memory")
@click.option("--clear_cache", is_flag=True, help="If set, the cache is cleared before training")
@click.option(
    "--list_size",
    type=int,
    default=16,
    help="The list size to use for the training (number of samples per graph in the batch)",
)
@click.option("--weight_decay", type=float, default=0.0, help="The weight decay to use for training")
@click.option("--n_configs_per_file", type=int, default=None, help="The number of configs to read per file")
@click.option("--n_configs_val", type=int, default=None, help="The number of configs to use for validation")
@click.option("--world_size", type=int, default=1, help="The number of GPUs to use for training")
@click.option("--max_train_steps", type=int, default=None, help="The maximum number of training steps per epoch")
@click.option("--dropout", type=float, default=0.25, help="The dropout to use for the network")
@click.option("--reload_configs", type=int, default=1, help="The number of epochs after which to reload the configs")
@click.option("--steps_per_batch", type=int, default=1, help="The number of steps per batch")
def main(**kwargs):
    # setup the distributed training
    mp.spawn(train_network, args=(kwargs,), nprocs=kwargs["world_size"], join=True)


if __name__ == "__main__":
    main()
