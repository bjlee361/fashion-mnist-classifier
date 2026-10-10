"""Figures: training curves, confusion matrix and misclassified examples."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files without needing a display
import matplotlib.pyplot as plt
import numpy as np
import torch

from .data import CLASS_NAMES, MEAN, STD


def plot_history(history: list[dict], out_path: str | Path, title: str = "") -> None:
    epochs = [h["epoch"] for h in history]
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 4))

    ax_loss.plot(epochs, [h["train_loss"] for h in history], marker="o", label="train")
    ax_loss.plot(epochs, [h["val_loss"] for h in history], marker="o", label="validation")
    ax_loss.set(xlabel="epoch", ylabel="cross-entropy loss", title="Loss")
    ax_loss.legend()

    ax_acc.plot(epochs, [100 * h["train_acc"] for h in history], marker="o", label="train")
    ax_acc.plot(epochs, [100 * h["val_acc"] for h in history], marker="o", label="validation")
    ax_acc.set(xlabel="epoch", ylabel="accuracy (%)", title="Accuracy")
    ax_acc.legend()

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def confusion_matrix(preds: torch.Tensor, labels: torch.Tensor, num_classes: int = 10) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(labels.tolist(), preds.tolist()):
        matrix[true, pred] += 1
    return matrix


def plot_confusion_matrix(matrix: np.ndarray, out_path: str | Path, title: str = "") -> None:
    # Normalize each row so cells read as "% of true class X predicted as Y".
    row_pct = 100 * matrix / matrix.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(row_pct, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set(xlabel="predicted", ylabel="true", title=title or "Confusion matrix (% of true class)")
    for i in range(row_pct.shape[0]):
        for j in range(row_pct.shape[1]):
            value = row_pct[i, j]
            if value >= 0.5:
                ax.text(j, i, f"{value:.0f}", ha="center", va="center",
                        color="white" if value > 50 else "black", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _unnormalize(image: torch.Tensor) -> np.ndarray:
    return (image.squeeze(0) * STD[0] + MEAN[0]).clamp(0, 1).numpy()


def plot_misclassified(dataset, preds: torch.Tensor, labels: torch.Tensor,
                       out_path: str | Path, n: int = 16) -> None:
    wrong = (preds != labels).nonzero().flatten()[:n].tolist()
    cols = 4
    rows = max(1, (len(wrong) + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(2.4 * cols, 2.6 * rows))
    for ax in np.atleast_1d(axes).flatten():
        ax.axis("off")
    for ax, idx in zip(np.atleast_1d(axes).flatten(), wrong):
        image, _ = dataset[idx]
        ax.imshow(_unnormalize(image), cmap="gray")
        true_name, pred_name = CLASS_NAMES[int(labels[idx])], CLASS_NAMES[int(preds[idx])]
        ax.set_title(f"true: {true_name}\npred: {pred_name}", fontsize=8)
    fig.suptitle("Misclassified test images")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
