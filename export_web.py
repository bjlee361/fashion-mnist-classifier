"""Export trained results and a model for the project website.

Writes two files into web/:
  site_data.json  - metrics, training history and test-set confusion matrix for every run in runs/
  model.json      - one model's weights (batch norm folded into the conv layers) for in-browser
                    inference, plus sample test images with PyTorch's own predictions so the
                    website can check its results match

Example:
    python export_web.py                 # demo model = runs/cnn/best.pt
    python export_web.py --demo-model mlp
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from fashion_classifier.data import CLASS_NAMES, MEAN, STD, get_test_set
from fashion_classifier.engine import evaluate
from fashion_classifier.models import count_parameters
from fashion_classifier.plots import confusion_matrix
from fashion_classifier.utils import load_checkpoint

ARG_KEYS = ("epochs", "batch_size", "optimizer", "lr", "weight_decay", "augment", "seed")


def b64_floats(tensor: torch.Tensor) -> str:
    array = np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype="<f4")
    return base64.b64encode(array.tobytes()).decode("ascii")


@torch.no_grad()
def export_layers(model: nn.Module) -> list[dict]:
    """Flatten the model into a simple list of layers the browser can run in order."""
    modules = [m for m in model.modules() if not list(m.children())]
    layers, i = [], 0
    while i < len(modules):
        m = modules[i]
        if isinstance(m, nn.Conv2d):
            if m.kernel_size != (3, 3) or m.stride != (1, 1) or m.padding != (1, 1):
                raise ValueError("The web demo only supports 3x3 convolutions with stride 1, padding 1")
            weight = m.weight.clone()
            bias = m.bias.clone() if m.bias is not None else torch.zeros(m.out_channels)
            if i + 1 < len(modules) and isinstance(modules[i + 1], nn.BatchNorm2d):
                bn = modules[i + 1]
                scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
                weight = weight * scale[:, None, None, None]
                bias = (bias - bn.running_mean) * scale + bn.bias
                i += 1
            layers.append({"type": "conv", "in": m.in_channels, "out": m.out_channels,
                           "w": b64_floats(weight), "b": b64_floats(bias)})
        elif isinstance(m, nn.Linear):
            layers.append({"type": "linear", "in": m.in_features, "out": m.out_features,
                           "w": b64_floats(m.weight), "b": b64_floats(m.bias)})
        elif isinstance(m, nn.ReLU):
            layers.append({"type": "relu"})
        elif isinstance(m, nn.MaxPool2d):
            if m.kernel_size not in (2, (2, 2)):
                raise ValueError("The web demo only supports 2x2 max pooling")
            layers.append({"type": "maxpool"})
        elif isinstance(m, nn.Flatten):
            layers.append({"type": "flatten"})
        elif isinstance(m, nn.Dropout):
            pass  # no-op at inference time
        else:
            raise ValueError(f"Layer type not supported by the web demo: {type(m).__name__}")
        i += 1
    return layers


@torch.no_grad()
def export_samples(model: nn.Module, test_set, per_class: int) -> list[dict]:
    targets = test_set.targets
    indices = []
    for c in range(len(CLASS_NAMES)):
        indices += (targets == c).nonzero().flatten()[:per_class].tolist()
    pixels = test_set.data[indices]  # uint8, N x 28 x 28
    x = ((pixels.float() / 255 - MEAN[0]) / STD[0]).unsqueeze(1)
    probs = torch.softmax(model(x), dim=1)
    return [
        {
            "index": idx,
            "label": int(targets[idx]),
            "pixels": base64.b64encode(pixels[n].numpy().tobytes()).decode("ascii"),
            "probs": [round(float(p), 6) for p in probs[n]],
        }
        for n, idx in enumerate(indices)
    ]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--demo-model", default="cnn", help="which run's model to put in the live demo")
    p.add_argument("--samples-per-class", type=int, default=3)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="web")
    args = p.parse_args()

    device = torch.device("cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    test_set = get_test_set(args.data_dir)
    loader = DataLoader(test_set, batch_size=256, shuffle=False)

    checkpoints = sorted(Path(args.runs_dir).glob("*/best.pt"))
    if not checkpoints:
        raise SystemExit(f"No checkpoints in {args.runs_dir}/. Train a model with train.py first.")

    runs = []
    for ckpt_path in checkpoints:
        run_dir = ckpt_path.parent
        model, ckpt = load_checkpoint(ckpt_path, device)
        result = evaluate(model, loader, nn.CrossEntropyLoss(), device, return_predictions=True)
        metrics_path, history_path = run_dir / "metrics.json", run_dir / "history.json"
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        history = json.loads(history_path.read_text()) if history_path.exists() else []
        train_args = metrics.get("args") or ckpt.get("args") or {}
        runs.append({
            "model": ckpt["model_name"],
            "parameters": count_parameters(model),
            "best_epoch": ckpt.get("epoch"),
            "val_acc": ckpt.get("val_acc"),
            "test_acc": result["acc"],
            "test_loss": result["loss"],
            "settings": {k: train_args.get(k) for k in ARG_KEYS},
            "history": history,
            "confusion": confusion_matrix(result["preds"], result["labels"]).tolist(),
        })
        print(f"{ckpt['model_name']:>4}: test accuracy {100 * result['acc']:.2f}%")

    (out_dir / "site_data.json").write_text(
        json.dumps({"classes": list(CLASS_NAMES), "runs": runs}, indent=1))

    demo_path = Path(args.runs_dir) / args.demo_model / "best.pt"
    model, ckpt = load_checkpoint(demo_path, device)
    payload = {
        "model": ckpt["model_name"],
        "mean": MEAN[0],
        "std": STD[0],
        "classes": list(CLASS_NAMES),
        "layers": export_layers(model),
        "samples": export_samples(model, test_set, args.samples_per_class),
    }
    (out_dir / "model.json").write_text(json.dumps(payload))

    size_mb = (out_dir / "model.json").stat().st_size / 1e6
    print(f"Wrote {out_dir}/site_data.json and {out_dir}/model.json ({size_mb:.1f} MB, "
          f"demo model: {ckpt['model_name']})")


if __name__ == "__main__":
    main()
