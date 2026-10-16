# Changelog

## October 2026

Protocol revision after a code review.

- Pretraining data matched. Both models now load ImageNet-1k-only weights by explicit tag. The October 2025 run used timm's default for the ViT, which is pretrained on ImageNet-21k, against an ImageNet-1k ResNet.
- Epoch selection moved to a validation split carved from the training set. The earlier run kept the best of ten test-set evaluations.
- Learning rate chosen per model and optimizer by a short sweep on the validation split (`--lr-sweep`), with linear warmup. The earlier run used one learning rate for both models, and the ResNet's poor result under SGD reflected that rather than the architecture.
- Weight decay now skips batch-norm as well as layer-norm parameters, so both models are treated the same way.
- Statistics: the gap is reported in percentage points with a t-interval. Cohen's d across seeds and the Wilcoxon test at five pairs are no longer reported; the first measures seed noise rather than effect size, and the second cannot reach 0.05 at that sample size.
- `analysis.py` shortened to the summary, calibration and RDM similarity.
- `all_results.csv` and `all_results.json` are written as the run goes, so every number in the docs can be regenerated.

## October 2025

First version: 2 models x 2 optimizers x 5 seeds, 10 epochs, Food-101.
