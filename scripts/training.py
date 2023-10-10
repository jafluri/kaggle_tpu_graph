import logging
import sys
from pathlib import Path

import click
import numpy as np
import torch
from torch import optim, nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from tpu_graph.data import TileDataset, LayoutDataset
from tpu_graph.networks import TPUGraphNetwork, SAGEConv, GPSConv

# from tpu_graph.training import evaluation
from tpu_graph.training.ltr.pairwise_losses import PairwiseDCGHingeLoss
from tqdm import tqdm
import os

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
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


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

    # load the dataset
    base_paths = [Path(p) for p in kwargs["data_path"]]

    # get the dataset class
    dataset_class = LayoutDataset if kwargs["layout_network"] else TileDataset

    logger.info("Loading the dataset for training")
    train_dataset = dataset_class(
        [base_path.joinpath("train") for base_path in base_paths],
        cache=kwargs["cache"],
        clear_cache=kwargs["clear_cache"],
        list_size=kwargs["list_size"],
        list_shuffle=True,
    )
    train_dataloader = train_dataset.get_dataloader(batch_size=kwargs["batch_size"])

    # logger.info("Loading the dataset for validation")
    # val_dataset = dataset_class(
    #     [base_path.joinpath("valid") for base_path in base_paths],
    #     cache=kwargs["cache"],
    #     list_size=1,
    #     clear_cache=kwargs["clear_cache"],
    # )
    # val_dataloader = val_dataset.get_dataloader(batch_size=32, shuffle=False, drop_last=False)
    #
    # logger.info("Loading the dataset for testing")
    # test_dataset = dataset_class(
    #     [base_path.joinpath("test") for base_path in base_paths],
    #     cache=kwargs["cache"],
    #     list_size=1,
    #     clear_cache=kwargs["clear_cache"],
    # )
    # test_dataloader = test_dataset.get_dataloader(batch_size=32, shuffle=False, drop_last=False)

    # we build a super simple network for starters
    logger.info("Building the network")
    input_dim = 159 if kwargs["layout_network"] else 165
    # the position embedding
    input_dim += 16

    # message_network = nn.Sequential(
    #     SAGEConv(256, 128), SAGEConv(128, 128), SAGEConv(128, 128), RetentiveAttention(128, 128)
    # )
    message_network = nn.Sequential(
        SAGEConv(256, 128),
        GPSConv(128, 128),
        SAGEConv(128, 128),
    )
    projection_network = nn.Linear(128, 1)

    network = TPUGraphNetwork(
        in_channels=input_dim,
        out_channels=256,
        message_network=message_network,
        projection_network=projection_network,
    )

    # restore the model if necessary
    if kwargs["restore_path"] is not None:
        logger.info("Restoring the model")
        network.load_state_dict(torch.load(kwargs["restore_path"]))

    # network to GPU
    network = network.to(rank)
    network = DDP(network, device_ids=[rank])

    # get the optimizer
    optimizer = optim.Adam(network.parameters(), lr=kwargs["learning_rate"], weight_decay=kwargs["weight_decay"])

    # get the total number of batches per epoch
    total = len(train_dataloader)
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
    loss_class = PairwiseDCGHingeLoss()
    batch_pad = torch.ones(kwargs["batch_size"]).long().to(rank) * train_dataset.list_size

    def loss_fn(pred, label):
        return loss_class(pred, label, batch_pad)

    # start the training loop
    logger.info("Starting the training loop")
    for epoch in range(kwargs["epochs"]):
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
            pred_runtimes = network(features, edge_index, lengths)
            loss = torch.mean(loss_fn(pred_runtimes, runtimes))

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # step the scheduler
            if scheduler is not None:
                scheduler.step()

            # log the loss to the logger
            if rank == 0:
                # get the total loss
                total_loss = dist.all_reduce(loss, op=dist.ReduceOp.SUM, async_op=False) / kwargs["world_size"]

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
            if kwargs["max_train_steps"] is not None and batch_idx >= kwargs["max_train_steps"]:
                break

    #     # save the network for this epoch
    #     logger.info("Saving the model")
    #     torch.save(network.state_dict(), save_path.joinpath(f"{wandb.run.name}_{epoch=}.pt"))
    #
    #     # validate the network
    #     if kwargs["layout_network"]:
    #         logger.info("Validating the network")
    #         avg_loss, avg_kendall = evaluation.evaluate_layout_network(
    #             network,
    #             val_dataloader,
    #             save_path.joinpath(f"{wandb.run.name}_{epoch=}_val.npz"),
    #         )
    #         # log everything
    #         wandb.log({"val_loss": avg_loss, "val_avg_kendall": avg_kendall}, commit=False)
    #         logger.info(f"Average kendall for epoch {epoch}: {avg_kendall}")
    #
    #         # test the network
    #         logger.info("Testing the network")
    #         avg_loss, avg_kendall = evaluation.evaluate_layout_network(
    #             network,
    #             test_dataloader,
    #             save_path.joinpath(f"{wandb.run.name}_{epoch=}_test.npz"),
    #         )
    #         # log everything
    #         wandb.log({"test_loss": avg_loss, "test_avg_kendall": avg_kendall})
    #         logger.info(f"Average kendall for epoch {epoch}: {avg_kendall}")
    #     else:
    #         logger.info("Validating the network")
    #         avg_loss, avg_slowdown = evaluation.evaluate_tile_network(
    #             network,
    #             val_dataloader,
    #             save_path.joinpath(f"{wandb.run.name}_{epoch=}_val.npz"),
    #         )
    #         # log everything
    #         wandb.log({"val_loss": avg_loss, "val_avg_slowdown": avg_slowdown}, commit=False)
    #         logger.info(f"Average slowdown for epoch {epoch}: {avg_slowdown}")
    #
    #         # test the network
    #         logger.info("Testing the network")
    #         avg_loss, avg_slowdown = evaluation.evaluate_tile_network(
    #             network,
    #             test_dataloader,
    #             save_path.joinpath(f"{wandb.run.name}_{epoch=}_test.npz"),
    #         )
    #         # log everything
    #         wandb.log({"test_loss": avg_loss, "test_avg_slowdown": avg_slowdown})
    #         logger.info(f"Average slowdown for epoch {epoch}: {avg_slowdown}")
    #
    #     # reshuffle the dataset
    #     logger.info("Reshuffling the dataset")
    #     train_dataset.reshuffle_indices()
    #     train_dataloader = train_dataset.get_dataloader(batch_size=kwargs["batch_size"])
    #
    # # save the model
    # logger.info("Saving the model")
    # torch.save(network.state_dict(), save_path.joinpath(f"{wandb.run.name}.pt"))


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
    "--layout_network",
    is_flag=True,
    help="If set, the layout network is trained, this changes the input dimension from 165 (tile network) "
    "to 159 (layout network)",
)
@click.option(
    "--list_size",
    type=int,
    default=16,
    help="The list size to use for the training (number of samples per graph in the batch)",
)
@click.option("--weight_decay", type=float, default=0.0, help="The weight decay to use for training")
@click.option("--max_train_steps", type=int, default=None, help="The maximum number of training steps")
@click.option("--world_size", type=int, default=1, help="The number of GPUs to use for training")
def main(**kwargs):
    # create a logger for the training
    logger = logging.getLogger("tpu_network.train.main")
    log_formatter = logging.Formatter(
        fmt="%(asctime)s %(name)10s %(levelname).3s   %(message)s ", datefmt="%y-%m-%d %H:%M:%S", style="%"
    )
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_formatter)
    logger.addHandler(stream_handler)
    logger.propagate = False
    logger.setLevel(logging.INFO)

    logger.info("Starting the training")

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
            "network_type": "Layout Network" if kwargs["layout_network"] else "Tile Network",
            "list_size": kwargs["list_size"],
        },
    )

    # print the run name
    logger.info(f"Run ID: {wandb.run.name}")

    # setup the distributed training
    logger.info("Setting up the distributed training")
    mp.spawn(train_network, args=(kwargs,), nprocs=kwargs["world_size"])


if __name__ == "__main__":
    main()
