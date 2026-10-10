"""Training and evaluation loops."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    """Run one pass over `loader`, updating weights. Returns mean loss and accuracy."""
    model.train()
    total_loss, correct, seen = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        # Weight by batch size so a smaller final batch doesn't skew the average.
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        seen += labels.size(0)
    return {"loss": total_loss / seen, "acc": correct / seen}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    return_predictions: bool = False,
) -> dict:
    """Compute mean loss and accuracy without updating weights.

    With `return_predictions=True`, also returns the predicted and true labels
    (CPU tensors) for building a confusion matrix.
    """
    model.eval()
    total_loss, correct, seen = 0.0, 0, 0
    all_preds, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += criterion(logits, labels).item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        seen += labels.size(0)
        if return_predictions:
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    result = {"loss": total_loss / seen, "acc": correct / seen}
    if return_predictions:
        result["preds"] = torch.cat(all_preds)
        result["labels"] = torch.cat(all_labels)
    return result


@torch.no_grad()
def predict_topk(model: nn.Module, image: torch.Tensor, k: int = 3) -> list[tuple[int, float]]:
    """Return the top-k (class index, probability) pairs for one 1x28x28 image."""
    model.eval()
    device = next(model.parameters()).device
    logits = model(image.unsqueeze(0).to(device))
    probs = torch.softmax(logits, dim=1).squeeze(0)
    values, indices = probs.topk(k)
    return [(int(i), float(v)) for i, v in zip(indices, values)]
