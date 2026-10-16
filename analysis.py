"""
Post-hoc analysis of a finished comparison run.

    python analysis.py

Reads fair_comparison_results/all_results.csv and the checkpoints written by
comparison.py, then produces:

- a summary table and the paired gap between the best ViT and ResNet configs
- test-set calibration (expected calibration error) for each best model
- how similar the two models' representations are (RDM correlation)
- a figure with accuracy per config, reliability diagrams and the two RDMs

Numbers are printed and also saved to fair_comparison_results/analysis.json.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torchvision
import torchvision.transforms as transforms
from scipy import stats
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

from comparison import Config

RESULTS = Config.RESULTS_DIR
DEVICE = Config.DEVICE


def load_runs():
    df = pd.read_csv(RESULTS / "all_results.csv")
    summary = df.groupby(["model", "optimizer"]).agg(
        test_mean=("test_acc", "mean"), test_sd=("test_acc", "std"),
        val_mean=("best_val_acc", "mean"), n=("seed", "count"),
        lr=("lr", "first"), weights=("weights", "first")).reset_index()
    return df, summary


def paired_gap(df, summary):
    """Best optimizer per model chosen on validation; gap on test, across matched seeds."""
    best = {m: summary[summary.model == m].sort_values("val_mean").iloc[-1].optimizer for m in Config.WEIGHTS}
    seeds = sorted(df.seed.unique())
    scores = {m: df[(df.model == m) & (df.optimizer == best[m])].set_index("seed").loc[seeds, "test_acc"].to_numpy()
              for m in Config.WEIGHTS}
    diff = scores["ViT-Small"] - scores["ResNet-34"]
    n = len(diff)
    out = {"best_optimizer": best, "seeds": seeds, "gap_mean_pp": float(diff.mean()),
           "gap_sd_pp": float(diff.std(ddof=1)) if n > 1 else None}
    if n > 1:
        lo, hi = stats.t.interval(0.95, n - 1, loc=diff.mean(), scale=stats.sem(diff))
        out["gap_ci95_pp"] = [float(lo), float(hi)]
        out["paired_t_p"] = float(stats.ttest_rel(scores["ViT-Small"], scores["ResNet-34"]).pvalue)
    return out, best


def test_loader():
    tf = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE + 32, Config.IMG_SIZE + 32)),
        transforms.CenterCrop(Config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    ds = torchvision.datasets.Food101(root="./data", split="test", download=False, transform=tf)
    return DataLoader(ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS), ds.classes


def load_best_checkpoint(df, model_name, optimizer):
    runs = df[(df.model == model_name) & (df.optimizer == optimizer)]
    seed = int(runs.sort_values("best_val_acc").iloc[-1].seed)
    path = RESULTS / f"{model_name}_{optimizer}_seed{seed}.pth"
    model = timm.create_model(Config.WEIGHTS[model_name], pretrained=False, num_classes=101)
    model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True)["model_state_dict"])
    return model.to(DEVICE).eval(), seed


@torch.no_grad()
def predict(model, loader):
    """Softmax probabilities, labels and pre-head features for the whole test set."""
    probs, labels, feats = [], [], []
    for images, y in loader:
        images = images.to(DEVICE)
        trunk = model.forward_features(images)
        probs.append(torch.softmax(model.forward_head(trunk), 1).cpu().numpy())
        feats.append(model.forward_head(trunk, pre_logits=True).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels), np.concatenate(feats)


def calibration(probs, labels, n_bins=15):
    conf = probs.max(1)
    correct = (probs.argmax(1) == labels).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece, bin_acc, bin_conf = 0.0, [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += abs(correct[m].mean() - conf[m].mean()) * m.mean()
            bin_acc.append(correct[m].mean())
            bin_conf.append(conf[m].mean())
    return float(ece), np.array(bin_conf), np.array(bin_acc)


def rdm(features):
    return 1 - np.corrcoef(features)


def rdm_similarity(rdm_a, rdm_b):
    mask = np.triu(np.ones_like(rdm_a, dtype=bool), k=1)
    return float(spearmanr(rdm_a[mask], rdm_b[mask])[0])


def main():
    df, summary = load_runs()
    print(summary.to_string(index=False))
    gap, best = paired_gap(df, summary)
    print(f"\nGap, ViT minus ResNet (best optimizer each, chosen on validation): {gap['gap_mean_pp']:.2f} pp")
    if "gap_ci95_pp" in gap:
        print(f"95% CI [{gap['gap_ci95_pp'][0]:.2f}, {gap['gap_ci95_pp'][1]:.2f}], paired t-test p = {gap['paired_t_p']:.2g}")
    print("The seeds measure training noise around one recipe, nothing more.")

    loader, classes = test_loader()
    results = {"summary": summary.to_dict(orient="records"), "paired_gap": gap, "models": {}}
    rng = np.random.RandomState(0)
    subset = rng.choice(len(loader.dataset), 500, replace=False)
    rdms, cal = {}, {}
    for model_name in Config.WEIGHTS:
        model, seed = load_best_checkpoint(df, model_name, best[model_name])
        probs, labels, feats = predict(model, loader)
        ece, bin_conf, bin_acc = calibration(probs, labels)
        cal[model_name] = (bin_conf, bin_acc)
        rdms[model_name] = rdm(feats[subset])
        acc = float((probs.argmax(1) == labels).mean() * 100)
        results["models"][model_name] = {"optimizer": best[model_name], "seed": seed, "test_acc": acc, "ece": ece}
        print(f"{model_name} + {best[model_name]} seed {seed}: test {acc:.2f}%, ECE {ece:.3f}")
    results["rdm_spearman_vit_vs_resnet"] = rdm_similarity(rdms["ViT-Small"], rdms["ResNet-34"])
    print(f"RDM similarity between the two models (Spearman, 500 test images): {results['rdm_spearman_vit_vs_resnet']:.3f}")

    (RESULTS / "analysis.json").write_text(json.dumps(results, indent=2, default=float))
    plot(summary, cal, rdms)
    print(f"Saved {RESULTS / 'analysis.json'} and {RESULTS / 'analysis.png'}")


def plot(summary, cal, rdms):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    ax = axes[0]
    labels = [f"{m}\n{o}" for m, o in zip(summary.model, summary.optimizer)]
    ax.bar(labels, summary.test_mean, yerr=summary.test_sd, capsize=3)
    ax.set_ylabel("test accuracy (%)")
    ax.set_title("accuracy per configuration, mean and sd over seeds")

    ax = axes[1]
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    for name, (conf, acc) in cal.items():
        ax.plot(conf, acc, marker="o", label=name)
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_title("reliability diagram")
    ax.legend()

    ax = axes[2]
    ax.imshow(rdms["ViT-Small"], cmap="viridis")
    ax.set_title("ViT-Small RDM, 500 test images")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(RESULTS / "analysis.png", dpi=200)


if __name__ == "__main__":
    main()
