# Bring Van Gogh Back to Life: Style Transfer & Classification

A deep learning project that (1) trains CNN classifiers to distinguish Vincent van Gogh paintings from other Post-Impressionist works, and (2) uses those classifiers as "judges" to guide a Neural Style Transfer pipeline that repaints ordinary photos in Van Gogh's style.

Final project for *Introduction to Deep Learning* (2026) — Nadav Amit, Guy Galanti, Yair Shapira.

📄 Full write-up: [Bring Van Gogh Back to Life - Report.pdf](<Bring Van Gogh Back to Life Style Transfer and Classification - Report.pdf>)

## Overview

**Part 1 — Classification.** Two ImageNet-pretrained backbones, VGG-19 and AlexNet, are fine-tuned (frozen backbone + trainable classifier head) on a 6,444-image Post-Impressionism dataset to perform binary classification: Van Gogh (15.6% of the data) vs. other artists. Class imbalance is handled with a weighted cross-entropy loss (`[1.0, 5.0]`), and hyperparameters (learning rate, batch size, optimizer) are tuned per-model with Optuna + Weights & Biases.

- AlexNet: Accuracy 0.933, F1 0.804, ROC-AUC 0.968 — higher recall, more false positives.
- VGG-19: Accuracy 0.927, F1 0.737, ROC-AUC 0.928 — higher precision (~83.5%), stricter and harder to fool.

VGG-19's stricter precision made it the preferred "judge" for Part 2.

**Part 2 — Style Transfer.** A classic Neural Style Transfer optimization (content loss + normalized Gram-matrix style loss + total-variation regularization) repaints photographs using five Van Gogh paintings as style references. Style transfer hyperparameters (learning rate, style/content/TV weights, per-layer weights) are tuned with Optuna to maximize the "Van Gogh probability" assigned by the Part 1 classifiers, then cross-evaluated with both classifiers acting as adversarial judges.

## Repository Structure

```
CNN Models/
  ALEXNET/            hyperparameter_search_AlexNet.ipynb, training_AlexNet.ipynb
  VGG/                hyperparameter_search_vgg.ipynb, training_vgg.ipynb
Style Transfers/
  AlexNet - Style Transfer/   style_transfer_hp_search_AlexNet.ipynb
  VGG - Style Transfer/       style_transfer_hp_search_VGG19.ipynb
Image Generating/
  ALEXNET - Image Generating/ part_c_AlexNet.ipynb
  VGG - Image Generating/     part_c_VGG19.ipynb
  Original Photos/            source photos used for style transfer
Original Photos/               (duplicate of the above, kept at repo root)
Bring Van Gogh Back to Life Style Transfer and Classification - Report.pdf
```

- **CNN Models** — hyperparameter search and full training of the two classifiers.
- **Style Transfers** — hyperparameter search for the style transfer optimization.
- **Image Generating** — generates stylized images with the tuned hyperparameters and scores them with both classifiers as judges.

## Requirements

- Python 3.x
- PyTorch & torchvision
- optuna
- wandb
- scikit-learn
- pandas
- Pillow (PIL)
- tqdm

```bash
pip install torch torchvision optuna wandb scikit-learn pandas pillow tqdm
```

A Weights & Biases account (`wandb login`) is needed to log hyperparameter search runs.

## Usage

The notebooks are meant to be run in order within each track:

1. `CNN Models/<MODEL>/hyperparameter_search_*.ipynb` — Optuna search for classifier hyperparameters.
2. `CNN Models/<MODEL>/training_*.ipynb` — full training and evaluation of the classifier with the best hyperparameters.
3. `Style Transfers/<MODEL> - Style Transfer/style_transfer_hp_search_*.ipynb` — Optuna search for style transfer hyperparameters, using the trained classifier as the objective.
4. `Image Generating/<MODEL> - Image Generating/part_c_*.ipynb` — generate stylized images from `Original Photos/` and score them with both classifiers.

## Key Findings

- Learning rate was the most influential classifier hyperparameter; Adam consistently outperformed SGD.
- VGG-19's depth makes it a more discerning (higher-precision) judge, while AlexNet is a more permissive, higher-recall classifier.
- Satisfying either classifier as a style-transfer judge required a very large style-loss weight (β ≈ 10⁷–10⁸) relative to content loss.
- The VGG-19 generator gravitated toward the fine, chaotic brushstrokes of *Dr. Paul Gachet*, while the AlexNet generator favored the bold swirls of *The Starry Night* — reflecting each architecture's receptive-field bias.
- GPU acceleration gave only a modest ~1.17x speedup over CPU due to an I/O-bound data pipeline (GPU utilization ~21%), not a compute bottleneck.

See the [report](<Bring Van Gogh Back to Life Style Transfer and Classification - Report.pdf>) for full methodology, hyperparameter tables, confusion matrices, and generated-image examples.
