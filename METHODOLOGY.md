# Methodology — Fair Architectural Comparison (ViT-Small vs ResNet-34)

> **Goal:** Provide a rigorous, reproducible, and minimal-bias experimental protocol for comparing Vision Transformer (ViT-Small) and ResNet-34 on Food-101.  
> This document records exact design choices, code locations, reproducibility measures, and analysis procedures so that results are verifiable and suitable as a computational baseline for future neural-alignment work.

---

## Table of contents
1. Experimental design (high-level)
2. Dataset
3. Models & parameter matching
4. Data augmentation (identical)
5. Training configuration & hyperparameters
6. Optimizers and weight-decay policy
7. Statistical analysis (multi-seed + tests)
8. Reproducibility & environment recording
9. Computational metrics and logging
10. Experimental workflow (exact steps)
11. Quality controls & validation
12. Limitations and mitigation
13. Extensions & future experiments
14. References (citations & BibTeX)
15. Reproducibility checklist (quick)

---

## 1. Experimental design (high-level)
- Question: When ViT and ResNet are compared under **identical** experimental conditions (same augmentation, same hyperparameter protocol, matched parameter counts, multi-seed runs), how much of the observed performance difference is attributable to architecture rather than experimental confounds?
- Approach: Controlled, factorial experiment:
  - 2 models × 2 optimizers × 5 seeds = 20 independent runs.
  - Save full metadata and checkpoints per run, aggregate results, and report paired statistical comparisons and effect sizes.

---

## 2. Dataset — Food-101
- **Dataset:** Food-101 (Bossard et al., ECCV 2014)
- **Details:** 101 classes, 101,000 images total (approx. 75,750 train / 25,250 test)
- **Preprocessing:** Resize & normalization to ImageNet stats; see Augmentation section.
- **Rationale:** Food-101 provides sufficient complexity (101 classes) and scale (100K images) for meaningful architectural comparison while remaining computationally tractable.


---

## 3. Models & parameter matching

### ViT-Small
- Source: `timm.create_model('vit_small_patch16_224', pretrained=True)`
- Parameters: ≈ **22,050,664**
- Basic config: 12 transformer blocks, 6 attention heads, patch=16, embed_dim=384

### ResNet-34
- Source: `timm.create_model('resnet34', pretrained=True)`
- Parameters: ≈ **21,797,672**

### Parameter matching
- Difference: ~1.1% (22.05M vs 21.80M). This close match reduces differences caused by parameter scale.
- Code locations:
  - model creation: `fine_tune_model()` in `comparison.py`
  - parameter counting: `param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)`

**Note:** Pretraining mismatch: ViT checkpoints typically come from ImageNet-21K→1K while the ResNet checkpoint is ImageNet-1K. We report this as a limitation and note it in the Limitations section.

---

## 4. Data augmentation (identical for both models)
All augmentations are shared between models (same pipeline object used for both).

### Training transforms
```python
train_transform = transforms.Compose([
    transforms.TrivialAugmentWide(),          # strong, parameter-free augmentation
    transforms.Resize((IMG_SIZE, IMG_SIZE)),  # IMG_SIZE = 224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])
```

### Test transforms

```python
test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])
```

**Design note:** TrivialAugmentWide is applied before resizing so augmentation occurs at higher resolution and the result is downsampled. This choice is explicitly identical for both models to remove augmentation bias.

Code: `get_dataloaders()` in `comparison.py`.

---

## 5. Training configuration & hyperparameters

### Shared hyperparameters

* Batch size: **512** (adjustable for GPU memory)
* Epochs: **10**
* Loss: `CrossEntropyLoss(label_smoothing=0.1)`
* LR schedule: `CosineAnnealingLR` with `T_max = NUM_EPOCHS`
* Mixed precision: enabled when CUDA available (`torch.cuda.amp`)

### Environment conditional

* AMP and GPU usage are enabled only when CUDA GPU present. Code checks `Config.DEVICE`.

**Exact fields (in code):** see `class Config` in `comparison.py`.

---

## 6. Optimizers and weight-decay handling

### Optimizer hyperparameters

* **AdamW**: lr = 3e-4, weight_decay = 0.05, betas = (0.9, 0.999)
* **SGD** : lr = 1e-2, weight_decay = 1e-4, momentum = 0.9

### Weight-decay policy

* Separate `decay` and `no_decay` parameter groups:

  * `no_decay`: parameters with `len(shape) == 1`, bias terms, and normalization layer params → `weight_decay = 0.0`
  * `decay`: other parameters → `weight_decay = configured value`
* Reason: Standard modern practice to avoid decaying normalization and bias params.

**Code:** `get_optimizer()` in `comparison.py`.

---

## 7. Statistical analysis (N = 5 seeds; paired tests)

### Seeds

```
SEEDS = [42, 101, 1337, 2024, 888]
```

Each run is independent (separate seed sets RNGs for NumPy, PyTorch, CUDA, and worker processes).

### Tests reported

1. **Paired t-test** (`scipy.stats.ttest_rel`) — primary test for mean difference under paired seeds.
2. **Wilcoxon signed-rank** (`scipy.stats.wilcoxon`) — nonparametric paired test (median difference), robust to non-normality.
3. **Cohen's d** (effect size) — compute using differences across matched seeds:
4. We compare each model’s best optimizer (by mean across seeds) — this is an optimizer ablation and not an unconstrained search.

```python
diff = np.array(vit_scores) - np.array(resnet_scores)
cohens_d = diff.mean() / diff.std(ddof=1)
```
**Note:** Here, we assumed diff.stf() > 0

4. **Confidence Intervals:** Report mean ± std and optionally 95% CI (bootstrap over seeds or t-distribution when N small).

### Interpretation

* Report p-values but focus on effect sizes and confidence intervals given small N.
* Do **not** claim definitive population-level statements beyond the experimental conditions.

**Code:** statistical block at the end of `comparison.py` (aggregation + `ttest_rel`, `wilcoxon`).

---

## 8. Reproducibility & environment recording (required)

### Deterministic seeding (code)

```python
class Config:
    @staticmethod
    def set_seed(seed: int):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
```

### Robust dataloader worker seeding (code)

```python
def worker_init_fn(worker_id):
    # Derive a per-worker seed from torch.initial_seed() for reproducible worker RNGs
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed + worker_id)
```

**Important notes:**

* Deterministic CUDA (`cudnn.deterministic=True`) may change runtime performance and make timing comparisons conservative. Mention this in results.
* Saving the exact environment (git commit, `pip freeze`) is mandatory.

### Commands to record environment (run before experiments)

```bash
# record git commit and environment
git rev-parse --short HEAD > results/commit_hash.txt
pip freeze > results/requirements.txt
nvidia-smi --query-gpu=name,memory.total --format=csv > results/gpu_info.txt
python -c "import torch,sys; print(torch.__version__, torch.cuda.is_available())" > results/pytorch_info.txt
```

Add these actions programmatically at experiment start to ensure metadata is saved with each run.

---

## 9. Computational metrics & logging

* **FLOPs**: measured with `ptflops.get_model_complexity_info()` and saved in run metadata.
* **Param count**: saved in run metadata.
* **Training time**: wall-clock time per epoch and total training time saved.
* **Memory usage**: `torch.cuda.max_memory_allocated()` recorded (if GPU).
* **Throughput**: images/sec measured during a training epoch.
* Save checkpoint meta and to aggregate CSV: `results/all_results.csv`.

---

## 10. Experimental workflow (exact steps)

1. `git clone ...` and check out commit recorded in README.
2. `pip install -r requirements.txt` (or use environment YAML).
3. Prepare results folder and save environment metadata (see Section 8).
4. Run `comparison.py` (script will also download Food-101 if necessary).

   * Example: `python comparison.py`
   * To reproduce a seed: `python comparison.py --seed 42`
5. Script trains (saves checkpoints) and writes per-run metadata to `results/`.
6. After all runs, run `analysis.py --results_dir results/` to produce summary table, plots, per-class metrics, and statistical tests.
7. Inspect `results/figures/` for confusion matrices and training curves.

---

## 11. Quality controls & validation

* **Sanity checks included (automated):**

  * Parameter counts within 5% for both models (assertion in script).
  * Train/test transforms are distinct; no test-time augmentation leakage.
  * Per-seed logs written; any run error halts and records stacktrace.
* **Validation checks (manual/automatic):**

  * Per-class accuracy and confusion matrix to identify class-specific failures.
  * Compare training vs test curves — anomalous spikes indicate issues.
  * Confirm best checkpoint loaded when computing test accuracy.

*Perform explicit per-class checks and ensure transformations are separated and no data-leakage.*

---

## 12. Limitations and mitigation

* **Single dataset:** Food-101 only. Results may not generalize across domains.
* **Pretraining mismatch:** ViT commonly uses ImageNet-21K→1K checkpoints while ResNet uses ImageNet-1K; this may bias results. We report this clearly and plan matched-pretraining controls.
* **Training length:** 10 epochs is brief; longer fine-tuning may alter optimizer behaviour (especially for ResNet + SGD).
* **No exhaustive hyperparameter sweep:** Standard hyperparameters are used for fairness; exhaustive tuning would be computationally expensive and must be applied equally to both models.
* **Deterministic mode affects timing:** Timings reported under deterministic settings may differ from non-deterministic runs.

---

## 13. Extensions & proposed experiments

* Longer runs: 50–100 epochs and early stopping.
* Learning-rate sweeps (per-optimizer) equally for both models.
* Add more datasets: CIFAR-100, ImageNet subset, iNaturalist.
* Architecture sweep: ViT-Tiny / ViT-Base and ResNet-50 / ResNet-101.
* Neural-alignment experiments: add spatial-latent heads and RDM/Brain-Score evaluations.
* Robustness testing: OOD datasets and adversarial attacks.

---

## 14. References (selected)

Papers and reviews that motivate our confound controls and reproducibility practices. Please cite these when discussing methodological design.

* Pineau, J., Vincent-Lamarre, P., Sinha, K., Larivière, V., Beygelzimer, A., d'Alché-Buc, F., Fox, E., & Larochelle, H. (2021). *Improving Reproducibility in Machine Learning Research: A Report from the NeurIPS 2019 Reproducibility Program.* Journal of Machine Learning Research. (Pineau et al., 2021)

* Li, C., Dakkak, A., Xiong, J., & Hwu, W.-m. (2019). *Challenges and Pitfalls of Machine Learning Evaluation and Benchmarking.* arXiv:1904.12437. (Li et al., 2019)

* Sculley, D., Holt, G., Golovin, D., Davydov, E., Phillips, T., Ebner, A., Chaudhary, V., Young, M., Crespo, J., & Dennison, D. (2015). *Hidden Technical Debt in Machine Learning Systems.* NeurIPS. (Sculley et al., 2015)

* Semmelrock, H., Kopeinik, S., Theiler, D., Ross-Hellauer, T., & Kowald, D. (2023). *Reproducibility in Machine Learning-Driven Research: Overview, Barriers, and Drivers.* arXiv:2307.10320. (Semmelrock et al., 2023)

* Yu, B. (2024). *After Computational Reproducibility: Scientific Reproducibility and Trustworthy AI.* Harvard Data Science Review. (Yu, 2024)

### References

```bibtex
@article{pineau2021improving,
  title = {Improving Reproducibility in Machine Learning Research: A Report from the NeurIPS 2019 Reproducibility Program},
  author = {Pineau, Joelle and Vincent-Lamarre, Philippe and Sinha, Koustuv and Larivi{\`e}re, Vincent and Beygelzimer, Alina and d'Alch{\'e}-Buc, Florence and Fox, Emily and Larochelle, Hugo},
  journal = {Journal of Machine Learning Research},
  volume = {22},
  number = {164},
  pages = {1--20},
  year = {2021},
  doi = {10.48550/arXiv.2003.12206},
  url = {https://jmlr.org/papers/v22/20-303.html},
  note = {Report from the NeurIPS 2019 reproducibility initiative}
}

@techreport{li2019challenges,
  title = {Challenges and Pitfalls of Machine Learning Evaluation and Benchmarking},
  author = {Li, Cheng and Dakkak, Abdul and Xiong, Jinjun and Hwu, Wen-mei W.},
  institution = {arXiv},
  year = {2019},
  doi = {10.48550/arXiv.1904.12437},
  url = {https://arxiv.org/abs/1904.12437},
  note = {Evaluation and benchmarking pitfalls; dataset leakage and reporting}
}

@inproceedings{sculley2015hidden,
  title = {Hidden Technical Debt in Machine Learning Systems},
  author = {Sculley, Daniel and Holt, Gary and Golovin, Daniel and Davydov, Eugene and Phillips, Todd and Ebner, Ankur and Chaudhary, Vinay and Young, Michael and Crespo, Jared and Dennison, Dan},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year = {2015},
  url = {https://papers.nips.cc/paper/2015/hash/5656-hidden-technical-debt-in-machine-learning-systems-abstract.html},
  note = {Canonical paper on engineering confounds in ML systems}
}

@article{semmelrock2023reproducibility,
  title = {Reproducibility in Machine Learning-Driven Research: Overview, Barriers, and Drivers},
  author = {Semmelrock, Harald and Kopeinik, Simone and Theiler, Dieter and Ross-Hellauer, Tony and Kowald, Dominik},
  journal = {arXiv preprint},
  year = {2023},
  doi = {10.48550/arXiv.2307.10320},
  url = {https://arxiv.org/abs/2307.10320},
  note = {Mini-survey reviewing reproducibility challenges}
}

@article{yu2024after,
  title = {After Computational Reproducibility: Scientific Reproducibility and Trustworthy AI},
  author = {Yu, Bin},
  journal = {Harvard Data Science Review},
  year = {2024},
  doi = {10.1162/99608f92.36c833d9},
  url = {https://hdsr.mitpress.mit.edu/pub/after-computational-reproducibility},
  note = {Perspective on scientific reproducibility and trustworthy AI}
}
```

---

## 15. Reproducibility checklist (quick)

* Single shared augmentation pipeline for all models (`get_dataloaders()`).
* Parameter counts and GFLOPs recorded per run.
* Multi-seed runs (SEEDS = [42,101,1337,2024,888]) and per-seed checkpoints saved.
* Deterministic seeding (`Config.set_seed()` + `worker_init_fn()`).
* Environment saved (`pip freeze`, git commit, GPU info).
* Paired statistical tests and effect size reported.
* Per-class metrics + confusion matrices available in `results/figures/`.
* Limitations documented (pretraining mismatch, dataset scope, epoch budget).
* `results/all_results.csv` and raw run metadata available for independent reanalysis.

---

## Contact & reproducibility support

Please email me at **[ambar13@u.nus.edu](mailto:ambar13@u.nus.edu)** or open an issue on the repository in case any clarification is needed.

---