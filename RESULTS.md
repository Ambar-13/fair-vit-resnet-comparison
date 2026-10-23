# Results

## Matched run, 23 October 2026

Both models with ImageNet-1k-only weights, the same pipeline, each at the learning rate a shared sweep chose for it, epochs picked on a validation split, test set evaluated once per run, five seeds. Files: `results/all_results.csv` (one row per run), `results/all_results.json` (with validation curves), `results/lr_sweep*.csv` (the grid and its two extensions), `results/chosen_learning_rates.json`.

| Model | Weights | Optimizer | Learning rate | Validation (mean) | Test accuracy, mean ± sd over 5 seeds |
|---|---|---|---|---|---|
| ViT-Small | vit_small_patch16_224.augreg_in1k | AdamW | 1e-4 | 82.67 | **87.11 ± 0.16** |
| ViT-Small | vit_small_patch16_224.augreg_in1k | SGD | 0.01 | 82.35 | 86.59 ± 0.07 |
| ResNet-34 | resnet34.a1_in1k | AdamW | 3e-3 | 77.24 | **81.95 ± 0.18** |
| ResNet-34 | resnet34.a1_in1k | SGD | 0.3 | 76.49 | 81.35 ± 0.09 |

Primary comparison, best optimizer per model chosen on validation (AdamW for both), paired by seed:

| Seed | ViT-Small | ResNet-34 |
|---|---|---|
| 42 | 87.33 | 82.24 |
| 101 | 86.99 | 81.92 |
| 1337 | 87.09 | 81.95 |
| 2024 | 86.95 | 81.73 |
| 888 | 87.19 | 81.92 |

**Gap, ViT minus ResNet: 5.16 percentage points, 95 percent CI [5.06, 5.27], paired t-test p = 2e-8.** The seeds measure training noise around one recipe; the interval says nothing about other datasets, budgets or pretraining.

### What changed from the October 2025 run, and by how much

| | October 2025 | Matched run | Change |
|---|---|---|---|
| ViT-Small, best | 90.05 (SGD, in21k weights) | 87.11 (AdamW, in1k weights) | −2.9 points: the ImageNet-21k pretraining |
| ResNet-34, best | 77.72 (AdamW, lr 3e-4) | 81.95 (AdamW, lr 3e-3) | +4.2 points: a learning rate chosen for it |
| ResNet-34, SGD | 46.69 (lr 0.01) | 81.35 (lr 0.3) | +34.7 points: the "collapse" was the learning rate |
| Gap | 12.33 | 5.16 | 7.2 of the 12.3 points were setup, not architecture |

Under matched pretraining data and matched tuning, ViT-Small still fine-tunes better than ResNet-34 on Food-101 at this budget, by about five points. Three fifths of the gap reported in October 2025 came from the ViT's extra pretraining data and the ResNet's learning rate. At its own rate the ResNet reaches 81.4 percent with SGD, within a point of its AdamW result, so the earlier instability claim does not stand.

### The learning-rate sweep

The main grid (AdamW 1e-4, 3e-4, 1e-3; SGD 1e-2, 3e-2, 1e-1; 3 epochs, seed 42) put both ResNet winners on its top edge, so the grid was extended one step at a time, for both models, until every winner was interior: AdamW 3e-3 then 1e-2, SGD 0.3 then 1.0. ViT accuracy fell steeply above its optimum (AdamW 3e-3: 60.6; 1e-2: 34.6; SGD 0.3 and 1.0 diverged). ResNet peaked at AdamW 3e-3 (75.8, against 72.3 one step higher) and SGD 0.3 (73.4, against 73.0). The two models want learning rates about ten to thirty times apart; the ViT is also far less tolerant of being above its optimum. That is a statement about optimisation under these pretraining recipes, not about representational quality.

### Caveats

- One dataset, one ten-epoch budget. Nothing here generalises beyond that.
- The pretraining recipes differ (AugReg for the ViT, RSB A1 for the ResNet) even though the data matches. Training both from scratch on identical data would remove this and is out of budget here.
- The sweep used one seed and three epochs; it is enough to avoid gross mis-tuning, not an exhaustive search.
- Runs landed on mixed GPU types, so training times are not comparable across runs and are not reported.

## October 2025 run, for the record

Not a matched comparison: the ViT used `vit_small_patch16_224.augreg_in21k_ft_in1k`, the reported accuracy is the best of ten test-set evaluations, and one learning rate per optimizer served both models. Checkpoints for seeds 42 and 101 are in `models/`.

| Model | Optimizer | Test acc, mean of 5 seeds | SD |
|---|---|---|---|
| ViT-Small (in21k then in1k) | SGD | 90.05 | 0.12 |
| ViT-Small (in21k then in1k) | AdamW | 89.87 | 0.14 |
| ResNet-34 (in1k) | AdamW | 77.72 | 0.27 |
| ResNet-34 (in1k) | SGD | 46.69 | 0.36 |

## Compute

ViT-Small 3.22 GFLOPs, ResNet-34 3.68 GFLOPs at 224 px (ptflops). The matched run used about 12 GPU-hours on the NUS SoC cluster, mostly on older cards.
