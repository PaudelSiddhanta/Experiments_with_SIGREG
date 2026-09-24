import argparse
import os
import time

import torch
from tqdm import tqdm

from datasets.cifar10 import build_ssl_loader
from models.ssl_model import SSLModel
from utils.config import load_config
from utils.seed import set_seed
from utils.logging import CSVLogger


def build_run_dir(config):

    base = config["experiment"]["output_dir"]
    name = config["experiment"]["name"]
    seed = config["experiment"]["seed"]

    run_dir = os.path.join(
        base,
        name,
        f"seed_{seed}",
    )

    checkpoint_dir = os.path.join(
        run_dir,
        "checkpoints",
    )

    os.makedirs(
        checkpoint_dir,
        exist_ok=True,
    )

    return run_dir


def save_checkpoint(
    path,
    epoch,
    model,
    optimizer,
    config,
):

    checkpoint = {
        "epoch": epoch,

        # Complete SSL model.
        "model": model.state_dict(),

        # Frozen representation encoder used
        # later by linear_probe.py.
        "encoder": model.encoder.state_dict(),

        "optimizer": optimizer.state_dict(),

        "config": config,
    }

    torch.save(
        checkpoint,
        path,
    )


def train(config):

    seed = config["experiment"]["seed"]

    set_seed(seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)
    print("Seed:", seed)

    method = config["method"]

    print(
        "Method:",
        config["experiment"]["name"],
    )

    print(
        "EMA:",
        method["use_ema"],
        "| stop-grad:",
        method["stop_gradient"],
        "| predictor:",
        method["use_predictor"],
        "| SIGReg:",
        method["use_sigreg"],
    )

    run_dir = build_run_dir(config)

    loader = build_ssl_loader(config)

    model = SSLModel(config).to(device)

    # Do NOT give frozen EMA target parameters
    # to optimizer.
    trainable_parameters = [
        p
        for p in model.parameters()
        if p.requires_grad
    ]

    optimizer = torch.optim.Adam(
        trainable_parameters,
        lr=config["train"]["lr"],
    )

    logger = CSVLogger(
        os.path.join(
            run_dir,
            "train.csv",
        ),
        [
            "epoch",
            "loss",
            "invariance_loss",
            "sigreg_loss",
            "lr",
            "batches",
            "seconds",
        ],
    )

    epochs = config["train"]["epochs"]

    checkpoint_every = config["train"][
        "checkpoint_every"
    ]

    max_batches = config["train"].get(
        "max_batches",
        None,
    )

    for epoch in range(
        1,
        epochs + 1,
    ):

        model.train()

        start_time = time.time()

        running_total = 0.0
        running_invariance = 0.0
        running_sigreg = 0.0

        num_batches = 0

        progress = tqdm(
            loader,
            desc=f"Epoch {epoch}/{epochs}",
        )

        for batch_idx, (
            images,
            _,
        ) in enumerate(progress):

            if (
                max_batches is not None
                and batch_idx >= max_batches
            ):
                break

            images = images.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            outputs = model(images)

            loss = outputs["loss"]

            loss.backward()

            optimizer.step()

            # Only performs anything for
            # EMA variants A/B.
            model.update_target()

            total_value = (
                outputs["loss"].item()
            )

            inv_value = (
                outputs[
                    "invariance_loss"
                ].item()
            )

            sig_value = (
                outputs[
                    "sigreg_loss"
                ].item()
            )

            running_total += total_value
            running_invariance += inv_value
            running_sigreg += sig_value

            num_batches += 1

            progress.set_postfix(
                loss=f"{total_value:.4f}",
                inv=f"{inv_value:.4f}",
                sig=f"{sig_value:.4f}",
            )

        average_total = (
            running_total / num_batches
        )

        average_invariance = (
            running_invariance
            / num_batches
        )

        average_sigreg = (
            running_sigreg
            / num_batches
        )

        elapsed = (
            time.time() - start_time
        )

        lr = optimizer.param_groups[0][
            "lr"
        ]

        print(
            f"Epoch {epoch}: "
            f"loss={average_total:.4f} "
            f"inv={average_invariance:.4f} "
            f"sig={average_sigreg:.4f} "
            f"time={elapsed:.1f}s"
        )

        logger.log({
            "epoch": epoch,
            "loss": average_total,
            "invariance_loss":
                average_invariance,
            "sigreg_loss":
                average_sigreg,
            "lr": lr,
            "batches": num_batches,
            "seconds": elapsed,
        })

        checkpoint_dir = os.path.join(
            run_dir,
            "checkpoints",
        )

        save_checkpoint(
            os.path.join(
                checkpoint_dir,
                "latest.pt",
            ),
            epoch,
            model,
            optimizer,
            config,
        )

        if (
            epoch % checkpoint_every == 0
            or epoch == epochs
        ):

            save_checkpoint(
                os.path.join(
                    checkpoint_dir,
                    f"epoch_{epoch}.pt",
                ),
                epoch,
                model,
                optimizer,
                config,
            )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
    )

    args = parser.parse_args()

    config = load_config(
        args.config
    )

    train(config)


if __name__ == "__main__":
    main()
