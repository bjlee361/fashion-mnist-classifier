# Fashion-MNIST Classifier: MLP vs. CNN in PyTorch

This project classifies 28×28 grayscale images of clothing into 10 categories (shirts, sneakers, bags and so on). It compares a fully connected network with a small convolutional network to show how much convolution helps on image data. The training pipeline uses a proper train/validation/test split, checkpointing, learning-rate scheduling, and diagnostic plots.

<!-- After training, put your confusion matrix here:
![CNN confusion matrix](assets/cnn_confusion_matrix.png)
-->

## Results

<!-- TODO: fill in from `python compare.py` after training both models -->

| Model | Parameters | Best epoch | Validation acc. | Test acc. |
|---|---:|---:|---:|---:|
| MLP | – | – | – | – |
| CNN | – | – | – | – |

<!-- TODO: add training curves, e.g.
![CNN training curves](assets/cnn_curves.png)
-->

**Observations** <!-- TODO: write 2–4 sentences of your own after looking at the plots. Prompts:
- Which classes does the model confuse most, and why might that be (look at misclassified.png)?
- Did the MLP overfit sooner than the CNN? Where do the train and validation curves split apart?
- What did augmentation change? -->

## Project structure

```
fashion_classifier/
  data.py      # dataset download, train/val/test split, normalization, augmentation
  models.py    # MLP baseline and SmallCNN (2 conv blocks + batch norm + dropout)
  engine.py    # training loop, evaluation loop, top-k prediction
  plots.py     # training curves, confusion matrix, misclassified examples
  utils.py     # seeding, device selection (CUDA / Apple MPS / CPU), checkpoints
train.py       # train a model, keep the best checkpoint by validation accuracy
evaluate.py    # per-class accuracy, confusion matrix, error analysis on the test set
predict.py     # top-k predictions for a test image or your own photo
compare.py     # summarize all runs as a Markdown table
tests/         # fast unit tests on random tensors (no download needed)
```

## Getting started

Requires Python 3.9+.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The dataset (~30 MB) downloads automatically into `data/` the first time you train.

```bash
# Train both models (the CNN is slower on a laptop CPU; use a GPU or Apple silicon if you have one)
python train.py --model mlp --epochs 15
python train.py --model cnn --epochs 15 --augment

# Detailed test-set analysis: per-class accuracy, confusion matrix, misclassified images
python evaluate.py runs/cnn/best.pt

# Predict a single image
python predict.py runs/cnn/best.pt --index 42
python predict.py runs/cnn/best.pt --image my_shoe.jpg --invert

# Results table for this README
python compare.py

# Unit tests
pytest
```

Each run writes `best.pt`, `metrics.json`, `history.json` and `curves.png` to `runs/<model>/`. Run `python train.py --help` to see all options (optimizer, learning rate, weight decay, batch size, seed, device).

## Design decisions

- **Validation split.** 10% of the official training set is held out with a fixed seed. The best epoch is chosen on validation accuracy, and the test set is scored only once at the end, so the reported test accuracy isn't tuned to the test data.
- **Correct normalization.** The pipeline uses the Fashion-MNIST mean and std (0.2860 / 0.3530) rather than the MNIST-digits values that are commonly copied into tutorials.
- **CNN architecture.** Two blocks of `conv → BN → ReLU → conv → BN → ReLU → max-pool` shrink each image from 28×28 to 7×7 with 64 channels. A dropout-regularized classifier head follows.
- **Optimization.** AdamW with weight decay and a cosine learning-rate schedule. SGD with Nesterov momentum is available through `--optimizer sgd`.
- **Correct averages.** Loss is weighted by batch size, so a smaller final batch doesn't skew the epoch average.

## Background

This started as a short intro-to-PyTorch exercise in a university AI course: a three-layer fully connected network trained for a few epochs. I rebuilt it as a standalone project. The rebuild adds a convolutional model and a comparison against the original approach, a held-out validation set with checkpointing, data augmentation, GPU/Apple-silicon support, error analysis, a command-line interface, and tests.

## Possible next steps

- Try a deeper network (e.g. a small ResNet) or mixed-precision training
- Average results over several random seeds and report the spread
- Export to ONNX and serve predictions from a small web demo

## Dataset

[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) by Zalando Research (MIT License): 60,000 training and 10,000 test images across 10 classes.
