"""Fast sanity checks that run on random tensors (no dataset download needed).

Run with:  pytest
"""

import pytest
import torch
from torch import nn

from fashion_classifier.engine import predict_topk
from fashion_classifier.models import MODELS, build_model
from fashion_classifier.utils import load_checkpoint, save_checkpoint


@pytest.mark.parametrize("name", list(MODELS))
def test_output_shape(name):
    model = build_model(name)
    logits = model(torch.randn(8, 1, 28, 28))
    assert logits.shape == (8, 10)


@pytest.mark.parametrize("name", list(MODELS))
def test_can_overfit_one_batch(name):
    """A working model + optimizer should drive loss down on a single fixed batch."""
    torch.manual_seed(0)
    model = build_model(name)
    images, labels = torch.randn(32, 1, 28, 28), torch.randint(0, 10, (32,))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()
    first_loss = criterion(model(images), labels).item()
    for _ in range(50):
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
    assert loss.item() < first_loss * 0.5


def test_checkpoint_round_trip(tmp_path):
    model = build_model("cnn").eval()
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, "cnn")

    restored, ckpt = load_checkpoint(path, torch.device("cpu"))
    x = torch.randn(4, 1, 28, 28)
    assert ckpt["model_name"] == "cnn"
    assert torch.allclose(model(x), restored(x))


def test_predict_topk_probabilities():
    model = build_model("mlp")
    top = predict_topk(model, torch.randn(1, 28, 28), k=3)
    assert len(top) == 3
    probs = [p for _, p in top]
    assert probs == sorted(probs, reverse=True)
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_unknown_model_name():
    with pytest.raises(ValueError):
        build_model("transformer")
