from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def _finalize(fig, save_path: str | Path) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curve(history: List[Dict[str, float]], save_path: str | Path) -> None:
    epochs = [item["epoch"] for item in history]
    train_loss = [item["train_loss"] for item in history]
    val_loss = [item["val_loss"] for item in history]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, train_loss, label="Train Loss", color="#1746A2", linewidth=2.2)
    ax.plot(epochs, val_loss, label="Val Loss", color="#FF731D", linewidth=2.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Conditional Diffusion Training Curve")
    ax.grid(alpha=0.25)
    ax.legend()
    _finalize(fig, save_path)


def plot_her_distribution(candidates: List[Dict], save_path: str | Path) -> None:
    delta_g = [item["predicted_properties"]["delta_g_h"] for item in candidates]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(delta_g, bins=12, color="#2A9D8F", alpha=0.82, edgecolor="white")
    ax.axvline(0.0, color="#D62828", linestyle="--", linewidth=2.0, label="Ideal ΔG_H = 0")
    ax.set_xlabel("Predicted ΔG_H (eV)")
    ax.set_ylabel("Count")
    ax.set_title("HER Activity Distribution of Generated 2D Materials")
    ax.grid(alpha=0.2)
    ax.legend()
    _finalize(fig, save_path)


def plot_stability_curve(candidates: List[Dict], save_path: str | Path) -> None:
    thermo = [item["predicted_properties"]["thermo_stability"] for item in candidates]
    kinetic = [item["predicted_properties"]["kinetic_stability"] for item in candidates]
    synth = [item["predicted_properties"]["synthesizability"] for item in candidates]
    x = np.arange(len(candidates))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, thermo, marker="o", linewidth=2, label="Thermo Stability", color="#264653")
    ax.plot(x, kinetic, marker="s", linewidth=2, label="Kinetic Stability", color="#E76F51")
    ax.plot(x, synth, marker="^", linewidth=2, label="Synthesizability", color="#F4A261")
    ax.set_xlabel("Generated Candidate")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Stability and Synthesizability of Top Candidates")
    ax.grid(alpha=0.2)
    ax.legend()
    _finalize(fig, save_path)


def plot_generated_structures(candidates: List[Dict], save_path: str | Path) -> None:
    top = candidates[:6]
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    axes = axes.flatten()
    for ax, material in zip(axes, top):
        positions = np.array(material["positions"])
        xs, ys = positions[:, 0], positions[:, 1]
        ax.scatter(xs, ys, s=120, color="#1D3557", alpha=0.9)
        for idx, element in enumerate(material["elements"]):
            ax.text(xs[idx] + 0.02, ys[idx] + 0.02, element, fontsize=9)
        ax.set_title(
            f"{material['formula']}\nΔG={material['predicted_properties']['delta_g_h']:.2f} eV",
            fontsize=10,
        )
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.grid(alpha=0.2)
    for ax in axes[len(top) :]:
        ax.axis("off")
    fig.suptitle("Top Generated 2D Material Structures", fontsize=14)
    _finalize(fig, save_path)
