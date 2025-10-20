# Results — Fair Architectural Comparison (ViT-Small vs ResNet-34)

## Executive summary
Under strictly controlled, reproducible conditions (identical augmentation, parameter matching, optimizer ablation, N=5 independent seeds), **ViT-Small outperforms ResNet-34 on Food-101**.  
- **Primary result:** ViT-Small (SGD) mean accuracy **90.05% ± 0.12%** vs ResNet-34 (AdamW) **77.72% ± 0.27%** (average over 5 seeds).  
- **Statistical evidence:** paired t-test p < 0.0001; Wilcoxon p = 0.0625; **Cohen’s d ≈ 65.22** (extremely large effect size).  
- **Interpretation (conservative):** Given the small seed count (N=5) but negligible cross-seed variance and non-overlapping 95% CIs, the observed ~12.3 percentage-point gap is robust in our experimental setting. We *do not* extrapolate beyond the exact conditions reported (dataset, pretraining, training budget, augmentations).

---

## Experiment summary (context)
- **Dataset:** Food-101 (101 classes; ~75.75k train / ~25.25k test).  
- **Models:** ViT-Small (≈22.05M params) and ResNet-34 (≈21.80M params).  
- **Design:** 2 models × 2 optimizers (AdamW, SGD) × 5 seeds = **20 runs**. Each run: 10 epochs, batch size and transforms identical, AMP on CUDA when available.  
- **Reporting:** per-run metadata and checkpoints saved to `results/`; aggregated CSV available at `results/all_results.csv`.

---

## Primary accuracy comparison (best optimizer per model)

| Model | Optimizer | Mean Accuracy (5 seeds) | Std Dev | 95% CI (approx.) |
|-------|-----------:|------------------------:|--------:|-----------------:|
| ViT-Small | **SGD** | **90.05%** | 0.119% | [89.88%, 90.22%] |
| ViT-Small | AdamW | 89.87% | 0.136% | [89.60%, 90.14%] |
| ResNet-34 | **AdamW** | 77.72% | 0.266% | [77.35%, 78.09%] |
| ResNet-34 | SGD | 46.69% | 0.358% | [45.86%, 47.52%] |

**Key observation:** ViT-Small achieves substantially higher absolute accuracy than ResNet-34 under the tested settings. ViT is also stable across optimizers; ResNet exhibits severe optimizer sensitivity (AdamW >> SGD).

---

## Optimizer ablation (summary & interpretation)

| Model | AdamW (mean) | SGD (mean) | Δ (SGD − AdamW) |
|------:|-------------:|-----------:|----------------:|
| ViT-Small | 89.87% | **90.05%** | **+0.18 pp** |
| ResNet-34 | **77.72%** | 46.69% | **−31.03 pp (collapse)** |

- **ViT:** small positive gain with SGD vs AdamW; low variance across seeds → robust optimization behavior.  
- **ResNet:** catastrophic underperformance with SGD at the chosen hyperparameters/training budget; AdamW produces reasonable performance.  
- **Implication:** identical optimizer choices can produce dramatically different outcomes across architectures; this is precisely why a fair, controlled comparison is necessary.

---

## Per-seed breakdown (best configurations shown)
> **ViT-Small (SGD)** — Seeds: 42,101,1337,2024,888  
> Accuracies: 89.93, 89.92, 90.16, 90.12, 90.14 — **Mean 90.05% ± 0.12%**

> **ResNet-34 (AdamW)** — Seeds: 42,101,1337,2024,888  
> Accuracies: 77.42, 77.59, 78.13, 77.80, 77.66 — **Mean 77.72% ± 0.27%**

(Full per-run table and metadata saved in `results/all_results.csv` and per-run JSON `meta` fields.)

---

## Statistical significance & effect size (paired analysis)
We compare ViT-Small (SGD) vs ResNet-34 (AdamW) using matched seeds.

- **Paired t-test:** t ≫ 0, **p < 0.0001**.  
- **Wilcoxon signed-rank:** p = 0.0625 (nonparametric; borderline given N=5).  
- **Cohen’s d:** ≈ **65.22** (effect >> conventional large threshold).  
- **Interpretation & caveats:**  
  - The t-test and very large Cohen’s d indicate a reproducible and practically important difference in this controlled setting.  
  - The Wilcoxon p (≈0.06) is marginally non-significant at α=0.05 — likely due to small sample size (N=5) and the paired nature; however, the parametric test and the extremely small variance support the conclusion that performance differs materially.  
  - We therefore report both tests and emphasize effect size and confidence intervals rather than relying on a single p-value.

---

## Learning dynamics & convergence
**Observed trends (empirical):**
- **ViT-Small (SGD):** smooth convergence, reaches ≈90% by epoch ~8.  
- **ViT-Small (AdamW):** fast early progress, slight plateau near 88–89%.  
- **ResNet-34 (AdamW):** steady increase across epochs up to ≈77–78%.  
- **ResNet-34 (SGD):** slow, limited convergence (≈12% → 46% across epochs), suggesting optimizer-architecture interaction or suboptimal LR/schedule for this condition.

**Actionable note:** ResNet+SGD may require different tuning (LR, warmup, longer training) to reach parity — see Limitations & next steps.

---

## Cross-seed stability (variance & CV)
- **Coefficient of variation (CV = std / mean):**
  - ViT-Small (SGD): **0.13%** — highly stable.
  - ViT-Small (AdamW): 0.15–0.20% — stable.
  - ResNet-34 (AdamW): **0.35%** — higher variance.
  - ResNet-34 (SGD): 0.36% — consistent failure (low variance around poor value).
- **Interpretation:** ViT exhibits ~2.7× lower stochastic sensitivity (CV) than ResNet in our protocol.

---

## Compute efficiency & throughput
- **GFLOPs (measured with ptflops):** ViT-Small ≈ 3.22 GFLOPs; ResNet-34 ≈ 3.68 GFLOPs (input 224×224).  
- **Training time (mean):** ViT runs ~1.0–1.05× slower per run (longer wall-clock) than ResNet in our setup.  
- **Accuracy per GFLOP:** ResNet shows slightly higher acc/GFLOP ratio in isolation, but **absolute accuracy** favors ViT by a large margin. For practical use, absolute performance + stability often outweighs minor FLOP efficiency differences.

---

## Per-class performance & confusion patterns
- **Best model (ViT-Small, SGD)**: mean per-class precision/recall ~0.90; confusion concentrated on visually similar classes (e.g., steak / prime_rib).  
- **ResNet failure case (SGD)**: many classes have low recall/precision; confusion more uniformly distributed — indicates poor feature alignment under these optimizer settings.

(Complete classification reports and confusion matrices are available in `results/figures/`.)

---

## Short discussion — causes & hypotheses
**Why does ResNet fail with SGD (hypotheses):**
1. **BatchNorm + SGD interaction:** BatchNorm's running statistics + high LR / short fine-tune schedule may destabilize learning.  
2. **LR and warmup:** ResNet may require a different LR schedule (lower LR or longer warmup) under SGD.  
3. **Training budget:** 10 epochs may be insufficient for ResNet+SGD fine-tuning from ImageNet weights on Food-101.

**Why ViT performs robustly:**
1. **LayerNorm & attention:** LayerNorm + residual attention blocks often create smoother optimization landscapes for SGD.  
2. **Pretraining differences:** ViT was loaded from ImageNet-21K→1K checkpoints in our runs (note this pretraining mismatch; see Limitations).  
3. **Augmentation compatibility:** TrivialAugmentWide + downsampling may particularly benefit transformer patch embeddings.

**Caveat:** hypotheses above are plausible given the data but require targeted follow-ups to validate.

---

## Limitations
1. **Single dataset:** Food-101 only. Results may not generalize across domains.
2. **Pretraining mismatch:** ViT commonly uses ImageNet-21K→1K checkpoints while ResNet uses ImageNet-1K; this may bias results. We report this clearly and plan matched-pretraining controls.
3. **Training length:** 10 epochs is brief; longer fine-tuning may alter optimizer behaviour (especially for ResNet + SGD).
4. **No exhaustive hyperparameter sweep:** Standard hyperparameters are used for fairness; exhaustive tuning would be computationally expensive and must be applied equally to both models.
5. **Deterministic mode affects timing:** Timings reported under deterministic settings may differ from non-deterministic runs.

---

## Recommended next steps
1. **Matched pretraining:** repeat with both models pretrained on the same corpus/checkpoint family.  
2. **Optimizer/LR sweeps:** small grid or random search for SGD LR/warmup for ResNet.  
3. **Longer fine-tuning:** run representative ResNet+SGD experiments for 50+ epochs with early stopping.  
4. **Cross-dataset replication:** run identical protocol on CIFAR-100 / ImageNet subset.  
5. **Representational analysis:** compute RDMs and Brain-Score style comparisons if neural data are available.
6. Test on more datasets: CIFAR-100, ImageNet subset, iNaturalist.
7. Architecture sweep: ViT-Tiny / ViT-Base and ResNet-50 / ResNet-101.
8. Neural-alignment experiments: add spatial-latent heads and RDM/Brain-Score evaluations.
9. Robustness testing: OOD datasets and adversarial attacks.

---

## Brief note on representational alignment
Preliminary RDM extractions (500 images) were computed for both models; initial visualization suggests ViT embeddings produce smoother, less anisotropic pairwise dissimilarities than ResNet under these settings — an observation worth exploring if connecting to neural alignment metrics.

---

## Conclusion
Under the exact, fully documented experimental protocol here (identical augmentation, matched parameter counts, multi-seed runs, optimizer ablation), **ViT-Small delivers substantially higher accuracy, much lower cross-seed variance, and greater optimizer robustness than ResNet-34 on Food-101.** The difference is statistically and practically large in this controlled setting. We report all raw artifacts (checkpoints, CSVs, logs, analysis scripts) so others can reproduce, probe alternative hyperparameters, or extend to neural-alignment evaluations.

---

## Data & code availability
All runs, checkpoints, aggregated CSVs, and analysis scripts are included in this repository under `results/` and `figures/`. To reproduce:  
```bash
python comparison.py        # runs training (see README for environment setup)
python analysis.py --results_dir results/
```