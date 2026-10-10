"""Show a trained model's top predictions for one image.

Use a test-set image by index, or your own photo of a clothing item.

Examples:
    python predict.py runs/cnn/best.pt --index 42
    python predict.py runs/cnn/best.pt --image my_sneaker.jpg
"""

from __future__ import annotations

import argparse

from PIL import Image, ImageOps

from fashion_classifier.data import CLASS_NAMES, build_transform, get_test_set
from fashion_classifier.engine import predict_topk
from fashion_classifier.utils import get_device, load_checkpoint


def load_custom_image(path: str, invert: bool):
    """Convert an arbitrary photo into a 1x28x28 tensor that looks like Fashion-MNIST.

    Fashion-MNIST items are light on a black background, so photos taken on
    a light background usually need --invert.
    """
    image = Image.open(path).convert("L")
    image = ImageOps.pad(image, (28, 28), color=255 if invert else 0)
    if invert:
        image = ImageOps.invert(image)
    return build_transform(augment=False)(image)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--index", type=int, help="index into the 10,000-image test set")
    source.add_argument("--image", help="path to your own image file")
    p.add_argument("--invert", action="store_true", help="invert a dark-on-light photo")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    model, _ = load_checkpoint(args.checkpoint, get_device(args.device))

    if args.index is not None:
        image, label = get_test_set(args.data_dir)[args.index]
        print(f"Test image #{args.index}, true label: {CLASS_NAMES[label]}")
    else:
        image = load_custom_image(args.image, args.invert)
        print(f"Image: {args.image}")

    for rank, (cls, prob) in enumerate(predict_topk(model, image, args.top_k), start=1):
        bar = "#" * round(prob * 30)
        print(f"  {rank}. {CLASS_NAMES[cls]:<12} {100 * prob:6.2f}%  {bar}")


if __name__ == "__main__":
    main()
