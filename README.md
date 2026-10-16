# ViT-Small vs ResNet-34 on Food-101, with matched pretraining

A controlled fine-tuning comparison of two 22M-parameter image models. The point of the repo is the protocol, not the winner.

## Status (16 October 2026)

The original run in this repo was **not** a fair comparison, and its README used to say it was. The ViT loaded ImageNet-21k weights (timm's default `vit_small_patch16_224` is `augreg_in21k_ft_in1k`) and the ResNet loaded ImageNet-1k weights (`resnet34.a1_in1k`). That difference in pretraining data, 14M images against 1.3M, is enough on its own to explain the 12-point gap the old README reported. Three more problems: the best epoch was selected on the test set, one learning rate was used for both architectures (so the ResNet's collapse under SGD was a tuning failure, not an architectural fact), and the statistics reported a Cohen's d of 65 computed on seed noise.

The code has been corrected. The matched rerun has **not** been done yet. [RESULTS.md](RESULTS.md) keeps the original numbers, labelled for what they are.

## What the corrected protocol does

| Control | How it is enforced in `comparison.py` |
|---|---|
| Matched pretraining data | Both models load ImageNet-1k-only weights by explicit tag: `vit_small_patch16_224.augreg_in1k` and `resnet34.a1_in1k`. The tag is written into every run's metadata. |
| Matched size | About 22M parameters each, 224 px input, FLOPs logged. |
| Identical data pipeline | Same augmentation (TrivialAugmentWide), same normalisation, same batches, same seeded validation split per seed. |
| Per-architecture learning rate | `--lr-sweep` runs the same short grid for every model and optimizer on a validation split and records the winners. One learning rate for both architectures favours whichever model likes that number. |
| No test-set selection | A 10 percent validation split is carved from the training set. The best-validation checkpoint is evaluated on the test set once. |
| Honest statistics | Gap in percentage points with a t-interval across seeds. No Cohen's d on seed noise; no Wilcoxon below six seeds, where its smallest possible p is 0.0625. |

## Running it

```bash
pip install -r requirements.txt
python comparison.py --lr-sweep        # 3-epoch sweep, one seed, writes chosen_learning_rates.json
python comparison.py                   # 2 models x 2 optimizers x 5 seeds, 10 epochs each
python analysis.py                     # figures, calibration, RDMs from the saved checkpoints
```

Outputs go to `fair_comparison_results/`: `all_results.csv`, `all_results.json` (with validation curves), one checkpoint per run, `lr_sweep.csv`.

## What this comparison can and cannot say

It can say which of two matched 22M models fine-tunes better on Food-101 in 10 epochs under a fair recipe. It cannot say anything about "ViT vs CNN" in general, and it says nothing about brain alignment. For the question of how much architecture matters for neural predictivity, the answer already exists at scale: Conwell et al. (NeurIPS 2021, mouse visual cortex) and Conwell et al. (Nature Communications 2024, 224 models, human visual cortex) both find that training data matters far more than architecture once training is matched.

## Files

- `comparison.py`: the experiment. Read the docstring at the top for the six controls.
- `analysis.py`: post-hoc analysis of saved checkpoints (ROC, calibration, RDMs, confusion matrices).
- `METHODOLOGY.md`: the protocol in detail.
- `RESULTS.md`: results, with the original unmatched run clearly labelled.
- `models/`: checkpoints from the original run, seeds 42 and 101 only, kept for reference.

## Author

Ambar, Computer Engineering, National University of Singapore.
