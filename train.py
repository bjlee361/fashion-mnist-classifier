"""Train a model on Fashion-MNIST and save the best checkpoint.

Examples:
    python train.py --model mlp --epochs 15
    python train.py --model cnn --epochs 15 --augment
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn

from fashion_classifier.data import get_loaders
from fashion_classifier.engine import evaluate, train_one_epoch
from fashion_classifier.models import MODELS, build_model, count_parameters
from fashion_classifier.plots import plot_history
from fashion_classifier.utils import get_device, save_checkpoint, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=MODELS, default="cnn")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    p.add_argument("--lr", type=float, default=None,
                   help="learning rate (default: 1e-3 for adam, 0.05 for sgd)")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--augment", action="store_true", help="random flips and crops on training images")
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--device", default="auto", help="auto, cpu, cuda or mps")
    p.add_argument("--num-workers", type=int, default=2)
    return p.parse_args()


def make_optimizer(name: str, params, lr: float | None, weight_decay: float):
    if name == "adam":
        return torch.optim.AdamW(params, lr=lr or 1e-3, weight_decay=weight_decay)
    return torch.optim.SGD(params, lr=lr or 0.05, momentum=0.9, nesterov=True,
                           weight_decay=weight_decay)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    run_dir = Path(args.out_dir) / args.model
    run_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = get_loaders(
        args.data_dir, args.batch_size, args.val_fraction, args.augment, args.seed, args.num_workers
    )

    model = build_model(args.model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(args.optimizer, model.parameters(), args.lr, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"Model: {args.model} ({count_parameters(model):,} parameters) on {device}")
    print(f"{'epoch':>5} | {'train loss':>10} {'train acc':>9} | {'val loss':>8} {'val acc':>8} | {'time':>5}")

    history, best_val_acc = [], 0.0
    checkpoint_path = run_dir / "best.pt"
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_stats = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_stats = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history.append({
            "epoch": epoch,
            "train_loss": train_stats["loss"], "train_acc": train_stats["acc"],
            "val_loss": val_stats["loss"], "val_acc": val_stats["acc"],
        })
        marker = ""
        if val_stats["acc"] > best_val_acc:
            best_val_acc = val_stats["acc"]
            save_checkpoint(checkpoint_path, model, args.model,
                            {"epoch": epoch, "val_acc": best_val_acc, "args": vars(args)})
            marker = "  *best"
        print(f"{epoch:>5} | {train_stats['loss']:>10.4f} {100 * train_stats['acc']:>8.2f}% | "
              f"{val_stats['loss']:>8.4f} {100 * val_stats['acc']:>7.2f}% | "
              f"{time.time() - start:>4.0f}s{marker}")

    # Final score: the best-on-validation weights, measured once on the untouched test set.
    best = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(best["state_dict"])
    test_stats = evaluate(model, test_loader, criterion, device)
    print(f"\nBest epoch {best['epoch']}: val acc {100 * best_val_acc:.2f}%, "
          f"test acc {100 * test_stats['acc']:.2f}%")

    metrics = {
        "model": args.model,
        "parameters": count_parameters(model),
        "best_epoch": best["epoch"],
        "val_acc": best_val_acc,
        "test_acc": test_stats["acc"],
        "test_loss": test_stats["loss"],
        "args": vars(args),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    plot_history(history, run_dir / "curves.png", title=f"{args.model.upper()} training")
    print(f"Saved checkpoint, metrics and curves to {run_dir}/")


if __name__ == "__main__":
    main()
