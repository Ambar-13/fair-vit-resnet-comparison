"""
Summary of a finished run, from the per-run JSON files alone (no torch needed).

    python report.py

Merges fair_comparison_results/runs/*.json into all_results.csv and
all_results.json, prints the per-configuration table and the paired gap
between the best ViT and the best ResNet configuration.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RESULTS_DIR = Path("fair_comparison_results")
MODELS = ["ViT-Small", "ResNet-34"]


def report(results_df: pd.DataFrame):
    """Summary table and the paired comparison of the best configuration per model."""
    summary = results_df.groupby(["model", "optimizer"]).agg(
        mean_test=("test_acc", "mean"), std_test=("test_acc", "std"),
        mean_val=("best_val_acc", "mean"), n=("seed", "count"),
        lr=("lr", "first"), weights=("weights", "first"),
    ).sort_values("mean_test", ascending=False).reset_index()
    print("\nSummary, mean over seeds")
    print(summary.to_string(index=False))

    # Best optimizer per model is chosen on validation accuracy, not test.
    best = {}
    for model_name in MODELS:
        sub = summary[summary.model == model_name]
        best[model_name] = sub.loc[sub.mean_val.idxmax(), "optimizer"]

    seeds = sorted(results_df.seed.unique())
    def scores(model_name):
        sub = results_df[(results_df.model == model_name) & (results_df.optimizer == best[model_name])]
        return sub.set_index("seed").loc[seeds, "test_acc"].to_numpy()

    vit, res = scores("ViT-Small"), scores("ResNet-34")
    diff = vit - res
    n = len(diff)
    print("\nPaired comparison, best optimizer per model chosen on validation")
    print(f"ViT-Small + {best['ViT-Small']}: {vit.round(2).tolist()}")
    print(f"ResNet-34 + {best['ResNet-34']}: {res.round(2).tolist()}")
    print(f"Gap (ViT minus ResNet): mean {diff.mean():.2f} pp, sd {diff.std(ddof=1):.2f} pp over {n} seeds")
    if n >= 2:
        ci = stats.t.interval(0.95, n - 1, loc=diff.mean(), scale=stats.sem(diff))
        p = stats.ttest_rel(vit, res).pvalue
        print(f"95% CI for the gap: [{ci[0]:.2f}, {ci[1]:.2f}] pp; paired t-test p = {p:.2g}")
    print("The seeds measure training noise around one recipe; the interval is about that noise.")
    if n < 6:
        print(f"Wilcoxon is not reported: with N={n} its smallest possible two-sided p is {2 / 2**n:.4f}.")




def main():
    runs = [json.loads(f.read_text()) for f in sorted((RESULTS_DIR / "runs").glob("*.json"))]
    if not runs:
        raise SystemExit("no runs/*.json yet")
    df = pd.DataFrame(runs)
    df.drop(columns=["val_curve"]).to_csv(RESULTS_DIR / "all_results.csv", index=False)
    (RESULTS_DIR / "all_results.json").write_text(json.dumps(runs, indent=2))
    print(f"{len(runs)} runs")
    report(df)


if __name__ == "__main__":
    main()
