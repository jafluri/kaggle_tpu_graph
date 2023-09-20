import logging
import sys
from pathlib import Path

import click
import wandb
import torch
from torch import optim, nn
from tqdm import tqdm

from tpu_graph.data import TileDataset
from tpu_graph.networks.tile_networks import TileNetwork
from tpu_graph.training import losses


@click.command()
@click.option("--data_path", type=click.Path(exists=True, dir_okay=True, file_okay=False), help="The path to the data")
@click.option(
    "--save_path",
    default=Path("./models"),
    type=click.Path(dir_okay=True, file_okay=False),
    help="The path to save the model",
)
@click.option("--learning_rate", type=float, default=0.001, help="The learning rate to use for training")
@click.option("--epochs", type=int, default=1, help="The number of epochs to train")
@click.option("--batch_size", type=int, default=16, help="The batch size to use for training")
@click.option("--cache", is_flag=True, help="If set, the dataset is cached in memory")
@click.option(
    "--mse",
    is_flag=True,
    help="If set, the mean squared error is used as loss function, " "otherwise the mean log squared error is used.",
)
def train_tile_network(**kwargs):
    # create a logger for the training
    logger = logging.getLogger("tile_network.train")
    log_formatter = logging.Formatter(
        fmt="%(asctime)s %(name)10s %(levelname).3s   %(message)s ", datefmt="%y-%m-%d %H:%M:%S", style="%"
    )
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_formatter)
    logger.addHandler(stream_handler)
    logger.propagate = False
    logger.setLevel(logging.INFO)
    logger.info("Starting training of the tile network")

    # Start with the wandb init
    logger.info("Starting wandb")
    wandb.init(
        project="TPU Graph",
        config={
            "learning_rate": kwargs["learning_rate"],
            "dataset": "Tiles Dataset of the TPU Graph Benchmark",
            "epochs": kwargs["epochs"],
            "batch_size": kwargs["batch_size"],
        },
    )

    # print the run name
    logger.info(f"Run ID: {wandb.run.name}")

    # load the dataset
    base_path = Path(kwargs["data_path"])

    logger.info("Loading the dataset for training")
    train_dataset = TileDataset(base_path.joinpath("train"), cache=kwargs["cache"])
    train_dataloader = train_dataset.get_dataloader(batch_size=kwargs["batch_size"])

    # logger.info("Loading the dataset for validation")
    # val_dataset = TileDataset(base_path.joinpath("valid"))
    # val_dataloader = val_dataset.get_dataloader(batch_size=kwargs["batch_size"])

    # we build a super simple network for starters
    logger.info("Building the network")
    network = TileNetwork(
        nn.Linear(165, 256),
        nn.SiLU(),
        nn.Linear(256, 256),
        nn.SiLU(),
        nn.Linear(256, 128),
        nn.SiLU(),
        nn.Linear(128, 128),
        nn.SiLU(),
        nn.Linear(128, 128),
        nn.SiLU(),
        nn.Linear(128, 1),
        nn.ReLU(),
    )

    # network to GPU
    network = network.to("cuda")

    # get the optimizer
    optimizer = optim.Adam(network.parameters(), lr=kwargs["learning_rate"])

    # start the training loop
    logger.info("Starting the training loop")
    for epoch in range(kwargs["epochs"]):
        pbar = tqdm(train_dataloader, postfix={"loss": 0})
        for batch_idx, (features, runtimes, edges, graphs) in enumerate(pbar):
            pred_runtimes = network.accumulate_runtime(features, edges, graphs)
            loss = losses.square_loss(pred=pred_runtimes, label=runtimes, log=not kwargs["mse"])

            # log the loss to wandb
            wandb.log({"loss": loss.item()})

            # log the loss to the logger
            pbar.set_postfix({"loss": loss.item()})

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # save the model
    logger.info("Saving the model")
    save_path = Path(kwargs["save_path"])
    save_path.mkdir(parents=True, exist_ok=True)
    torch.save(network.state_dict(), save_path.joinpath(f"{wandb.run.name}.pt"))


if __name__ == "__main__":
    train_tile_network()
