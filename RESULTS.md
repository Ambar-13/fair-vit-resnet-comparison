# Results

## Matched-pretraining run (the corrected protocol)

Not yet run. When it is, the table below gets filled from `fair_comparison_results/all_results.csv` and the sweep results from `lr_sweep.csv`. Expect the gap to shrink substantially once the ViT no longer carries ImageNet-21k weights, and expect the ResNet to train normally under SGD once its learning rate is chosen rather than inherited.

| Model | Weights | Optimizer | LR (from sweep) | Val acc | Test acc (mean over 5 seeds) |
|---|---|---|---|---|---|
| ViT-Small | vit_small_patch16_224.augreg_in1k | AdamW | | | |
| ViT-Small | vit_small_patch16_224.augreg_in1k | SGD | | | |
| ResNet-34 | resnet34.a1_in1k | AdamW | | | |
| ResNet-34 | resnet34.a1_in1k | SGD | | | |

## Original run (October 2025), kept for the record

This run is **not** a matched comparison. Read the caveats before the numbers.

- ViT-Small used `vit_small_patch16_224.augreg_in21k_ft_in1k` (pretrained on ImageNet-21k, then ImageNet-1k). ResNet-34 used `resnet34.a1_in1k` (ImageNet-1k only).
- The reported accuracy is the best of ten test-set evaluations, one per epoch. There was no validation split.
- One learning rate per optimizer for both architectures (AdamW 3e-4, SGD 1e-2), no warmup, batch 512, 10 epochs, cosine schedule.
- Checkpoints for seeds 42 and 101 are in `models/`. The per-seed CSV for the other three seeds was not committed; the numbers below are from the run log.

| Model | Optimizer | Test acc, mean of 5 seeds | SD |
|---|---|---|---|
| ViT-Small (in21k then in1k) | SGD | 90.05 | 0.12 |
| ViT-Small (in21k then in1k) | AdamW | 89.87 | 0.14 |
| ResNet-34 (in1k) | AdamW | 77.72 | 0.27 |
| ResNet-34 (in1k) | SGD | 46.69 | 0.36 |

Per seed (42, 101, 1337, 2024, 888): ViT+SGD 89.93, 89.92, 90.16, 90.12, 90.14; ResNet+AdamW 77.42, 77.59, 78.13, 77.80, 77.66.

Gap, ViT+SGD minus ResNet+AdamW: 12.33 percentage points, 95 percent CI [12.10, 12.57], paired t-test p = 1e-8.

What that gap means: with these two checkpoints and this recipe, the result is almost deterministic across seeds. It does not measure architecture, because the two models did not see the same pretraining data, and it does not measure the ResNet's capability under SGD, because the SGD learning rate was never tuned for it. The old README reported Cohen's d = 65 and called it a very large effect; that number is the mean gap divided by the seed-to-seed standard deviation of the gap (0.19 points), so it measures how little training noise there is, not how large the effect is. Wilcoxon p = 0.0625 was reported as borderline; it is the smallest value the test can return with five pairs.

## Compute

ViT-Small about 3.2 GFLOPs, ResNet-34 about 3.7 GFLOPs at 224 px (ptflops). Wall-clock per run was within 5 percent of each other on the original hardware.
