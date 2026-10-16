# ViT-Small vs ResNet-34 on Food-101, with matched pretraining

A controlled fine-tuning comparison of two 22M-parameter image models. The point of the repo is the protocol: both models get the same pretraining data, the same pipeline, a learning rate chosen the same way, and a validation split for every selection step, so that whatever gap remains is about the models and not about the setup.

## Protocol

| Control | How `comparison.py` enforces it |
|---|---|
| Same pretraining data | Both models load ImageNet-1k-only weights by explicit tag: `vit_small_patch16_224.augreg_in1k` and `resnet34.a1_in1k`. The tag is written into every run's metadata. |
| Same size | About 22M parameters each, 224 px input, FLOPs logged. |
| Same data pipeline | Same augmentation (TrivialAugmentWide), normalisation, batches and seeded validation split per seed. |
| Learning rate per model | `--lr-sweep` runs the same short grid for every model and optimizer on the validation split and records the winners. |
| No test-set selection | A 10 percent validation split is carved from the training set. The best-validation checkpoint is evaluated on the test set once. |
| Statistics | The gap in percentage points with a t-interval across five seeds. |

Details are in [METHODOLOGY.md](METHODOLOGY.md).

## Running it

```bash
pip install -r requirements.txt
python comparison.py --lr-sweep        # 3-epoch sweep, one seed, writes chosen_learning_rates.json
python comparison.py                   # 2 models x 2 optimizers x 5 seeds, 10 epochs each
python analysis.py                     # calibration, RDM similarity, figure
```

Outputs go to `fair_comparison_results/`: `all_results.csv`, `all_results.json` (with validation curves), one checkpoint per run, `lr_sweep.csv`.

## Results

See [RESULTS.md](RESULTS.md). The matched run is scheduled; the table there is filled from `all_results.csv` when it finishes. An earlier, unmatched run from October 2025 is reported on the same page for reference.

## Scope

This answers one question: which of two matched 22M models fine-tunes better on Food-101 in ten epochs under a fair recipe. It does not speak to "ViT vs CNN" in general or to brain alignment. For how much architecture matters to neural predictivity, see Conwell et al. (NeurIPS 2021) and Conwell et al. (Nature Communications 2024).

## Files

- `comparison.py`: the experiment; the docstring lists the controls.
- `analysis.py`: post-hoc analysis of saved checkpoints.
- `METHODOLOGY.md`, `RESULTS.md`, `CHANGELOG.md`.
- `models/`: checkpoints from the October 2025 run, seeds 42 and 101.

## Revision note

October 2026: the protocol was revised after a review of the code. The October 2025 run had loaded ImageNet-21k weights for the ViT and ImageNet-1k weights for the ResNet, which is not a matched comparison; it also selected epochs on the test set and used one learning rate for both models. The code now enforces each control above, and the earlier numbers are kept in RESULTS.md with that context. The full list of changes is in [CHANGELOG.md](CHANGELOG.md).

## Author

Ambar, Computer Engineering, National University of Singapore.
