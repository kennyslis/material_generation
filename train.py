import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.material_dataset import MaterialDataset
from models.diffusion_model import CrystalDiffusionModel
from utils.vis import plot_loss_curve


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_fn(batch):
    output = {}
    for key in batch[0]:
        output[key] = torch.stack([item[key] for item in batch], dim=0)
    return output


def evaluate(model, loader, device, property_weights):
    model.eval()
    losses = []
    metrics = {"delta_g_mae": [], "stability_mae": [], "synth_mae": []}
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            result = model(
                node_features=batch["node_features"],
                positions=batch["positions"],
                mask=batch["mask"],
                condition=batch["condition"],
                targets=batch["targets"],
                property_weights=property_weights.to(device),
            )
            losses.append(result.metrics["loss"])
            for key in metrics:
                metrics[key].append(result.metrics[key])
    return {
        "loss": float(np.mean(losses)),
        "delta_g_mae": float(np.mean(metrics["delta_g_mae"])),
        "stability_mae": float(np.mean(metrics["stability_mae"])),
        "synth_mae": float(np.mean(metrics["synth_mae"])),
    }


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    train_dataset = MaterialDataset(Path(args.output_dir) / "dataset", split="train", num_samples=args.train_samples)
    val_dataset = MaterialDataset(Path(args.output_dir) / "dataset", split="val", num_samples=args.val_samples)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = CrystalDiffusionModel(node_dim=4, condition_dim=4, hidden_dim=args.hidden_dim, diffusion_steps=args.diffusion_steps)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    property_weights = torch.tensor([1.4, 0.9, 0.9, 0.8], dtype=torch.float32)

    history = []
    best_val = float("inf")
    checkpoint_dir = Path(args.output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            result = model(
                node_features=batch["node_features"],
                positions=batch["positions"],
                mask=batch["mask"],
                condition=batch["condition"],
                targets=batch["targets"],
                property_weights=property_weights.to(device),
            )
            result.total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(result.metrics["loss"])

        scheduler.step()
        val_metrics = evaluate(model, val_loader, device, property_weights)
        train_loss = float(np.mean(epoch_losses))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_metrics["loss"], **val_metrics})

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": vars(args),
                    "history": history,
                },
                checkpoint_dir / "best_model.pt",
            )

        if epoch % args.log_interval == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d} | train={train_loss:.4f} | val={val_metrics['loss']:.4f} | "
                f"ΔG MAE={val_metrics['delta_g_mae']:.4f}"
            )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": vars(args),
            "history": history,
        },
        checkpoint_dir / "last_model.pt",
    )

    history_path = Path(args.output_dir) / "results" / "training_history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("w", encoding="utf-8") as fp:
        json.dump(history, fp, indent=2)
    plot_loss_curve(history, Path(args.output_dir) / "results" / "loss_curve.png")
    print(f"Training finished. Best val loss: {best_val:.4f}")


def build_parser():
    parser = argparse.ArgumentParser(description="Train a conditional diffusion model for 2D material generation")
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--diffusion-steps", type=int, default=60)
    parser.add_argument("--train-samples", type=int, default=320)
    parser.add_argument("--val-samples", type=int, default=96)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--cpu", action="store_true")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    train(args)
