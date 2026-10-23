# Methodology

The question: when ViT-Small and ResNet-34 are fine-tuned on Food-101 under the same recipe, with the same pretraining data, the same size, and a learning rate chosen the same way for each, how large is the difference in test accuracy, and how stable is it across seeds?

The question this does not answer: whether "transformers beat CNNs". One dataset, one budget, two models.

## 1. Models and weights

| Model | timm tag | Pretraining | Parameters |
|---|---|---|---|
| ViT-Small/16 | `vit_small_patch16_224.augreg_in1k` | ImageNet-1k only, AugReg recipe (Steiner et al. 2021) | 21.7M |
| ResNet-34 | `resnet34.a1_in1k` | ImageNet-1k only, RSB A1 recipe (Wightman et al. 2021) | 21.8M |

The tags are explicit because timm's defaults are not matched: without a tag, `vit_small_patch16_224` loads `augreg_in21k_ft_in1k`, which was pretrained on ImageNet-21k. Both recipes are modern and heavily augmented, so neither model is handicapped by an old training setup. The tag is recorded in every run's metadata so the comparison can be audited from the CSV alone.

A stricter version would train both from scratch on the same data. That is out of budget here and is listed under extensions.

## 2. Data

Food-101: 101 classes, 75,750 training images, 25,250 test images. The official test split is used only for the final evaluation. A validation split of 10 percent of the training images is carved out with a fixed seed per run; it is the same split for both architectures within a seed.

## 3. Augmentation and preprocessing

Train: TrivialAugmentWide, resize to 224 x 224, ImageNet normalisation. Validation and test: resize to 256, centre crop 224, same normalisation. Identical for both models. The train resize distorts aspect ratio; both models get the same distortion.

## 4. Optimisation

- Optimizers: AdamW (weight decay 0.05) and SGD with Nesterov momentum 0.9 (weight decay 1e-4).
- No weight decay on biases or any normalisation parameter. The check covers `norm`, `bn` and `downsample.1`, so batch norm in the ResNet and layer norm in the ViT are treated the same way.
- Schedule: linear warmup for one epoch, then cosine to zero, stepped per batch.
- Batch 512, 10 epochs, mixed precision on CUDA, label smoothing 0.1.

### Learning rate per architecture

`python comparison.py --lr-sweep` trains every (model, optimizer, lr) in the grid below for 3 epochs on seed 42 and keeps the learning rate with the best validation accuracy. The grid and budget are the same for both models; that is the fairness condition. The chosen values are written to `fair_comparison_results/chosen_learning_rates.json` and read by the main run.

| Optimizer | Grid |
|---|---|
| AdamW | 1e-4, 3e-4, 1e-3 |
| SGD | 1e-2, 3e-2, 1e-1 |

If a winner sits on the top edge of the grid, the grid is extended one step upward for both models (`slurm/sweep_up.sh`, then `sweep_up2.sh`) and the choice is remade over every sweep file by `choose_lr.py`, which refuses to write while any winner is still on the edge. In the matched run the ResNet needed two extensions (final rates AdamW 3e-3, SGD 0.3); the ViT's winners were interior from the start (AdamW 1e-4, SGD 0.01).

If no sweep file exists the script falls back to the original unmatched values (AdamW 3e-4, SGD 1e-2) and prints a warning. Do not report results from the fallback.

## 5. Model selection and evaluation

Validation accuracy is measured after every epoch. The checkpoint with the best validation accuracy is kept and evaluated on the test set once. The test set is never used to pick an epoch, a learning rate, or an optimizer.

The best optimizer per model (for the headline paired comparison) is chosen on mean validation accuracy across seeds, not on test accuracy.

## 6. Seeds and statistics

Five seeds: 42, 101, 1337, 2024, 888. Each seed sets NumPy, PyTorch and CUDA RNGs, the dataloader workers, and the validation split.

Reported: per-configuration mean and SD of test accuracy; the paired gap between the best ViT configuration and the best ResNet configuration in percentage points, with a 95 percent t-interval and a paired t-test p-value.

Not reported, and why:
- Cohen's d across seeds. With training noise of about 0.2 points, any real gap gives a d in the tens. It measures determinism, not effect size.
- Wilcoxon signed-rank with five pairs. Its smallest possible two-sided p is 0.0625, so it cannot reach 0.05 and carries no information at this N. It is printed only from six seeds up.

What the seeds measure: optimisation noise around one recipe. They do not tell you whether the ranking survives a different dataset, a longer budget, or different pretraining.

## 7. Compute

FLOPs are measured with ptflops at 224 px. Training time is logged per run. These are context, not a result.

## 8. Reproducibility

- `comparison.py --lr-sweep`, then `comparison.py`, then `analysis.py`.
- Every run writes its weight tag, learning rate, best epoch, validation curve and test accuracy to `all_results.csv` and `all_results.json`.
- Checkpoints are saved per run as `{model}_{optimizer}_seed{seed}.pth` with the metadata inside.
- Deterministic cuDNN is on; timings under it are slower than in production.

## 9. Limitations

- One dataset and one fine-tuning budget. Nothing here generalises beyond that.
- Pretraining recipes differ (AugReg vs RSB A1) even though the data matches. From-scratch training on identical data would remove this.
- The sweep is short (3 epochs) and uses one seed. It is enough to avoid gross mis-tuning, which is what sank the original ResNet+SGD run; it is not an exhaustive search.
- Parameter matching does not match pretraining compute.

## 10. Extensions

- From-scratch training on ImageNet-1k with the same recipe for both.
- A second dataset with a different image statistics profile (iNaturalist, or a texture-heavy set).
- Longer budgets (30 and 100 epochs) to see whether the gap moves.
- ConvNeXt-T and Swin-T as a second matched pair.

## References

- Steiner et al. 2021, How to train your ViT? Data, augmentation, and regularization in vision transformers (AugReg).
- Wightman, Touvron, Jégou 2021, ResNet strikes back: an improved training procedure in timm (RSB).
- Dosovitskiy et al. 2021, An image is worth 16x16 words.
- Bossard et al. 2014, Food-101.
