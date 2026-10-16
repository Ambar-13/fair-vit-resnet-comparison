# Results

## Matched run (current protocol)

Scheduled. The table is filled from `fair_comparison_results/all_results.csv` and the sweep results from `lr_sweep.csv` when the run finishes.

| Model | Weights | Optimizer | LR (from sweep) | Val acc | Test acc (mean over 5 seeds) |
|---|---|---|---|---|---|
| ViT-Small | vit_small_patch16_224.augreg_in1k | AdamW | | | |
| ViT-Small | vit_small_patch16_224.augreg_in1k | SGD | | | |
| ResNet-34 | resnet34.a1_in1k | AdamW | | | |
| ResNet-34 | resnet34.a1_in1k | SGD | | | |

## October 2025 run, for reference

This run predates the current protocol; see CHANGELOG.md. In particular the ViT used `vit_small_patch16_224.augreg_in21k_ft_in1k` (ImageNet-21k then ImageNet-1k) while the ResNet used `resnet34.a1_in1k` (ImageNet-1k), the reported accuracy is the best of ten test-set evaluations, and one learning rate per optimizer was used for both models (AdamW 3e-4, SGD 1e-2, no warmup). Checkpoints for seeds 42 and 101 are in `models/`.

| Model | Optimizer | Test acc, mean of 5 seeds | SD |
|---|---|---|---|
| ViT-Small (in21k then in1k) | SGD | 90.05 | 0.12 |
| ViT-Small (in21k then in1k) | AdamW | 89.87 | 0.14 |
| ResNet-34 (in1k) | AdamW | 77.72 | 0.27 |
| ResNet-34 (in1k) | SGD | 46.69 | 0.36 |

Per seed (42, 101, 1337, 2024, 888): ViT+SGD 89.93, 89.92, 90.16, 90.12, 90.14; ResNet+AdamW 77.42, 77.59, 78.13, 77.80, 77.66.

Gap, ViT+SGD minus ResNet+AdamW: 12.33 percentage points, 95 percent CI [12.10, 12.57]. The gap is stable across seeds, but because the pretraining data differed it should not be read as a difference between the architectures.

## Compute

ViT-Small about 3.2 GFLOPs, ResNet-34 about 3.7 GFLOPs at 224 px (ptflops).
