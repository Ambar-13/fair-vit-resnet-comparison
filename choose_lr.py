"""
Pick the final learning rate per model and optimizer from every sweep file,
and write chosen_learning_rates.json.

    python choose_lr.py

Reads fair_comparison_results/lr_sweep*.csv (the main grid and any steps above
it), takes the best validation accuracy per (model, optimizer), and refuses to
write if a winner still sits on the top edge of the grid that was tried, since
that means the grid has to be extended again.
"""

import json
from pathlib import Path

import pandas as pd

RESULTS = Path("fair_comparison_results")


def main():
    files = sorted(RESULTS.glob("lr_sweep*.csv"))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    best = df.loc[df.groupby(["model", "optimizer"]).best_val_acc.idxmax()]
    top = df.groupby("optimizer").lr.max()
    print(best.to_string(index=False))
    on_edge = [(r.model, r.optimizer, r.lr) for _, r in best.iterrows() if r.lr == top[r.optimizer]]
    if on_edge:
        raise SystemExit(f"still on the top edge of the grid, extend it first: {on_edge}")
    chosen = {}
    for _, r in best.iterrows():
        chosen.setdefault(r.model, {})[r.optimizer] = float(r.lr)
    (RESULTS / "chosen_learning_rates.json").write_text(json.dumps(chosen, indent=2))
    print("wrote", json.dumps(chosen))


if __name__ == "__main__":
    main()
