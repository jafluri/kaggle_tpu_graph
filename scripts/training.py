import logging
import sys
from pathlib import Path

import click
import torch
from torch import optim, nn
from tpu_graph.data import TileDataset, LayoutDataset
from tpu_graph.networks import TPUGraphNetwork
from tpu_graph.training import losses, evaluation
from tqdm import tqdm

import wandb


@click.command()
@click.option("--data_path", type=click.Path(exists=True, dir_okay=True, file_okay=False), help="The path to the data")
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
    "--cosine_annealing_tmax",
    type=int,
    default=0,
    help="The number of steps for the cosine annealing, if 0, no cosine annealing is used",
)
@click.option("--epochs", type=int, default=1, help="The number of epochs to train")
@click.option("--batch_size", type=int, default=16, help="The batch size to use for training")
@click.option("--cache", is_flag=True, help="If set, the dataset is cached in memory")
@click.option(
    "--mse",
    is_flag=True,
    help="If set, the mean squared error is used as loss function, otherwise the mean log squared error is used.",
)
@click.option(
    "--layout_network",
    is_flag=True,
    help="If set, the layout network is trained, this changes the input dimension from 165 (tile network) "
    "to 159 (layout network)",
)
@click.option(
    "--p_update_path",
    type=float,
    default=1.0,
    help="The probability to update the path in the network for a given graph during training",
)
@click.option(
    "--fast_eval",
    is_flag=True,
    help="If set, we calculate the longest path only once and use it for all iterations of the same graph "
    "during the evaluation",
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
            "cosine_annealing_tmax": kwargs["cosine_annealing_tmax"],
            "network_type": "Layout Network" if kwargs["layout_network"] else "Tile Network",
            "loss_type": "MSE" if kwargs["mse"] else "Log MSE",
            "p_update_path": kwargs["p_update_path"],
        },
    )

    # print the run name
    logger.info(f"Run ID: {wandb.run.name}")

    # load the dataset
    base_path = Path(kwargs["data_path"])

    # get the dataset class
    dataset_class = LayoutDataset if kwargs["layout_network"] else TileDataset

    logger.info("Loading the dataset for training")
    train_dataset = dataset_class(base_path.joinpath("train"), cache=kwargs["cache"])
    train_dataloader = train_dataset.get_dataloader(batch_size=kwargs["batch_size"])

    logger.info("Loading the dataset for validation")
    val_dataset = dataset_class(base_path.joinpath("valid"), cache=kwargs["cache"])
    val_dataloader = val_dataset.get_dataloader(batch_size=kwargs["batch_size"], shuffle=False)

    logger.info("Loading the dataset for testing")
    test_dataset = dataset_class(base_path.joinpath("test"), cache=kwargs["cache"])
    test_dataloader = test_dataset.get_dataloader(batch_size=kwargs["batch_size"], shuffle=False)

    # we build a super simple network for starters
    logger.info("Building the network")
    input_dim = 159 if kwargs["layout_network"] else 165
    network = TPUGraphNetwork(
        nn.Linear(input_dim, 256),
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

    # restore the model if necessary
    if kwargs["restore_path"] is not None:
        logger.info("Restoring the model")
        network.load_state_dict(torch.load(kwargs["restore_path"]))

    # network to GPU
    network = network.to("cuda")

    # get the optimizer
    optimizer = optim.Adam(network.parameters(), lr=kwargs["learning_rate"])

    # get the scheduler
    scheduler = None
    if kwargs["cosine_annealing_tmax"] > 0:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, kwargs["cosine_annealing_tmax"], eta_min=0.000001)

    # create the saving path
    save_path = Path(kwargs["save_path"])
    save_path.mkdir(parents=True, exist_ok=True)

    # start the training loop
    logger.info("Starting the training loop")
    for epoch in range(kwargs["epochs"]):
        logger.info(f"Starting epoch {epoch}")
        pbar = tqdm(train_dataloader, postfix={"loss": 0})
        for batch_idx, (features, runtimes, edges, graphs) in enumerate(pbar):
            pred_runtimes = network.accumulate_runtime(features, edges, graphs, kwargs["p_update_path"])
            loss = losses.square_loss(pred=pred_runtimes, label=runtimes, log=not kwargs["mse"])

            # log the loss to wandb
            wandb.log({"loss": loss.item()})

            # log the loss to the logger
            pbar.set_postfix({"loss": loss.item()})

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # step the scheduler
            if scheduler is not None:
                wandb.log({"lr": scheduler.get_last_lr()[0]})
                scheduler.step()

        # reset the scheduler
        if scheduler is not None:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, kwargs["cosine_annealing_tmax"], eta_min=0.000001
            )

        # save the network for this epoch
        logger.info("Saving the model")
        torch.save(network.state_dict(), save_path.joinpath(f"{wandb.run.name}_{epoch=}.pt"))

        # validate the network
        if kwargs["layout_network"]:
            logger.info("Validating the network")
            avg_loss, avg_kendall = evaluation.evaluate_layout_network(
                network,
                val_dataloader,
                save_path.joinpath(f"{wandb.run.name}_{epoch=}_val.npz", fast_eval=kwargs["fast_eval"]),
            )
            # log everything
            wandb.log({"val_loss": avg_loss})
            wandb.log({"avg_kendall": avg_kendall})
            logger.info(f"Average kendall for epoch {epoch}: {avg_kendall}")

            # test the network
            logger.info("Testing the network")
            avg_loss, avg_kendall = evaluation.evaluate_layout_network(
                network,
                test_dataloader,
                save_path.joinpath(f"{wandb.run.name}_{epoch=}_test.npz", fast_eval=kwargs["fast_eval"]),
            )
            # log everything
            wandb.log({"test_loss": avg_loss})
            wandb.log({"avg_kendall": avg_kendall})
            logger.info(f"Average kendall for epoch {epoch}: {avg_kendall}")
        else:
            logger.info("Validating the network")
            avg_loss, avg_slowdown = evaluation.evaluate_tile_network(
                network,
                val_dataloader,
                save_path.joinpath(f"{wandb.run.name}_{epoch=}_val.npz", fast_eval=kwargs["fast_eval"]),
            )
            # log everything
            wandb.log({"val_loss": avg_loss})
            wandb.log({"avg_slowdown": avg_slowdown})
            logger.info(f"Average slowdown for epoch {epoch}: {avg_slowdown}")

            # test the network
            logger.info("Testing the network")
            avg_loss, avg_slowdown = evaluation.evaluate_tile_network(
                network,
                test_dataloader,
                save_path.joinpath(f"{wandb.run.name}_{epoch=}_test.npz", fast_eval=kwargs["fast_eval"]),
            )
            # log everything
            wandb.log({"test_loss": avg_loss})
            wandb.log({"avg_slowdown": avg_slowdown})
            logger.info(f"Average slowdown for epoch {epoch}: {avg_slowdown}")

    # save the model
    logger.info("Saving the model")
    torch.save(network.state_dict(), save_path.joinpath(f"{wandb.run.name}.pt"))


if __name__ == "__main__":
    train_tile_network()
