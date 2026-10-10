"""Evaluate a trained checkpoint on the test set in detail.

Prints per-class accuracy and saves a confusion matrix and a grid of
misclassified images next to the checkpoint.

Example:
    python evaluate.py runs/cnn/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from torch import nn
from torch.utils.data import DataLoader

from fashion_classifier.data import CLASS_NAMES, get_test_set
from fashion_classifier.engine import evaluate
from fashion_classifier.plots import confusion_matrix, plot_confusion_matrix, plot_misclassified
from fashion_classifier.utils import get_device, load_checkpoint


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = get_device(args.device)
    model, ckpt = load_checkpoint(args.checkpoint, device)
    test_set = get_test_set(args.data_dir)
    loader = DataLoader(test_set, batch_size=256, shuffle=False)

    result = evaluate(model, loader, nn.CrossEntropyLoss(), device, return_predictions=True)
    preds, labels = result["preds"], result["labels"]
    print(f"Model: {ckpt['model_name']}  |  test accuracy {100 * result['acc']:.2f}%  |  "
          f"test loss {result['loss']:.4f}\n")

    matrix = confusion_matrix(preds, labels)
    print(f"{'class':<12} {'accuracy':>8}   most often confused with")
    for i, name in enumerate(CLASS_NAMES):
        row = matrix[i].copy()
        acc = row[i] / row.sum()
        row[i] = -1
        confused = CLASS_NAMES[row.argmax()]
        print(f"{name:<12} {100 * acc:>7.1f}%   {confused}")

    out_dir = args.checkpoint.parent
    plot_confusion_matrix(matrix, out_dir / "confusion_matrix.png",
                          title=f"{ckpt['model_name'].upper()} confusion matrix (% of true class)")
    plot_misclassified(test_set, preds, labels, out_dir / "misclassified.png")
    print(f"\nSaved confusion_matrix.png and misclassified.png to {out_dir}/")


if __name__ == "__main__":
    main()
