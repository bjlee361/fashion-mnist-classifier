"""Model definitions: a fully connected baseline and a small CNN."""

from __future__ import annotations

from torch import nn


class MLP(nn.Module):
    """Fully connected network on flattened 28x28 images."""

    def __init__(
        self,
        hidden: tuple[int, ...] = (256, 128),
        dropout: float = 0.2,
        num_classes: int = 10,
    ):
        super().__init__()
        layers: list[nn.Module] = [nn.Flatten()]
        in_features = 28 * 28
        for width in hidden:
            layers += [nn.Linear(in_features, width), nn.ReLU(), nn.Dropout(dropout)]
            in_features = width
        layers.append(nn.Linear(in_features, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """Two 3x3 conv layers with batch norm, then 2x2 max pooling (halves H and W)."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SmallCNN(nn.Module):
    """Two convolutional blocks (28 -> 14 -> 7) followed by a small classifier head."""

    def __init__(self, num_classes: int = 10, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(_conv_block(1, 32), _conv_block(32, 64))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


MODELS = {"mlp": MLP, "cnn": SmallCNN}


def build_model(name: str, **kwargs) -> nn.Module:
    try:
        return MODELS[name](**kwargs)
    except KeyError:
        raise ValueError(f"Unknown model '{name}'. Choose from: {', '.join(MODELS)}") from None


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
