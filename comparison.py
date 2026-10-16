"""
Fair architectural comparison: ViT-Small vs ResNet-34 on Food-101.

What "fair" means in this script, and how each point is enforced:

1. Matched pretraining data. Both models load ImageNet-1k-only weights, named
   explicitly below. The timm defaults are NOT matched (the ViT default is
   ImageNet-21k then 1k; the ResNet default is 1k only), which is why the
   tags are spelled out and recorded in every run's metadata.
2. Matched parameter count (about 22M each) and matched input size.
3. Identical data pipeline and augmentation.
4. Per-architecture learning rate, chosen by the same short sweep on a
   validation split (run with --lr-sweep). One learning rate for both
   architectures is not fairness; it favours whichever model happens to like
   that number.
5. Epoch selection on a validation split carved from the training set, never
   on the test set. The test set is evaluated once, with the best-validation
   checkpoint.
6. Several seeds, with the difference reported in percentage points with a
   confidence interval. No Cohen's d on seed noise.

Usage:
    python comparison.py --lr-sweep            # pick LR per model and optimizer
    python comparison.py                       # full run with chosen LRs
    python comparison.py --seeds 42 101        # subset of seeds
"""

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from scipy import stats
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

try:
    from ptflops import get_model_complexity_info
except ImportError:
    get_model_complexity_info = None


class Config:
    SEEDS: List[int] = [42, 101, 1337, 2024, 888]
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_EPOCHS: int = 10
    WARMUP_EPOCHS: int = 1
    BATCH_SIZE: int = 512
    IMG_SIZE: int = 224
    NUM_WORKERS: int = 8
    VAL_FRACTION: float = 0.1
    RESULTS_DIR: Path = Path("fair_comparison_results")
    CHOSEN_LR_FILE: Path = RESULTS_DIR / "chosen_learning_rates.json"

    # Explicit weight tags. Both are ImageNet-1k only.
    WEIGHTS = {
        "ViT-Small": "vit_small_patch16_224.augreg_in1k",
        "ResNet-34": "resnet34.a1_in1k",
    }

    # Learning-rate grid for the sweep, per optimizer. The sweep runs every
    # value for every model with one seed and a short budget, and keeps the
    # one with the best validation accuracy.
    LR_GRID = {
        "AdamW": [1e-4, 3e-4, 1e-3],
        "SGD": [1e-2, 3e-2, 1e-1],
    }
    SWEEP_EPOCHS: int = 3
    WEIGHT_DECAY = {"AdamW": 0.05, "SGD": 1e-4}
    MOMENTUM = 0.9

    # Used when no sweep file exists. These are the original values and are
    # not matched to either architecture; run --lr-sweep before trusting them.
    FALLBACK_LR = {"AdamW": 3e-4, "SGD": 1e-2}

    @staticmethod
    def set_seed(seed: int):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed + worker_id)


def get_dataloaders(seed: int, batch_size=Config.BATCH_SIZE, img_size=Config.IMG_SIZE):
    """Food-101 with a seeded validation split carved from the training set.

    The validation loader uses the test transform (no augmentation). The same
    split is used for both architectures in a given seed.
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.TrivialAugmentWide(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize,
    ])

    train_full = torchvision.datasets.Food101(root="./data", split="train", download=True, transform=train_transform)
    val_full = torchvision.datasets.Food101(root="./data", split="train", download=True, transform=eval_transform)
    test_dataset = torchvision.datasets.Food101(root="./data", split="test", download=True, transform=eval_transform)

    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(train_full))
    n_val = int(len(train_full) * Config.VAL_FRACTION)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    train_loader = DataLoader(Subset(train_full, train_idx), batch_size=batch_size, shuffle=True,
                              num_workers=Config.NUM_WORKERS, pin_memory=True, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(Subset(val_full, val_idx), batch_size=batch_size, shuffle=False,
                            num_workers=Config.NUM_WORKERS, pin_memory=True, worker_init_fn=worker_init_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=Config.NUM_WORKERS, pin_memory=True, worker_init_fn=worker_init_fn)

    num_classes = len(train_full.classes)
    print(f"Data: {len(train_idx)} train, {len(val_idx)} val, {len(test_dataset)} test ({num_classes} classes)")
    return train_loader, val_loader, test_loader, num_classes


def is_no_decay_param(name: str, param: torch.Tensor) -> bool:
    """Biases and all normalisation parameters get no weight decay.

    timm ResNets name batch norm 'bn1', 'bn2', 'downsample.1'; ViTs name layer
    norm 'norm', 'norm1', 'norm2'. Checking only for 'norm' decayed the
    ResNet's batch-norm weights and not the ViT's, an asymmetry the old code had.
    """
    if param.ndim <= 1:
        return True
    if name.endswith(".bias"):
        return True
    return any(tok in name for tok in ("norm", "bn", "downsample.1"))


def get_optimizer(model, optimizer_name: str, lr: float):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (no_decay if is_no_decay_param(name, param) else decay).append(param)
    params = [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": Config.WEIGHT_DECAY[optimizer_name]},
    ]
    if optimizer_name == "AdamW":
        return optim.AdamW(params, lr=lr)
    if optimizer_name == "SGD":
        return optim.SGD(params, lr=lr, momentum=Config.MOMENTUM, nesterov=True)
    raise ValueError(f"Unknown optimizer: {optimizer_name}")


def warmup_cosine(optimizer, num_epochs: int, warmup_epochs: int, steps_per_epoch: int):
    """Linear warmup for warmup_epochs, then cosine to zero, stepped per batch."""
    total = num_epochs * steps_per_epoch
    warm = warmup_epochs * steps_per_epoch

    def factor(step):
        if step < warm:
            return (step + 1) / warm
        progress = (step - warm) / max(1, total - warm)
        return 0.5 * (1 + np.cos(np.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, factor)


def build_model(model_name: str, num_classes: int) -> nn.Module:
    tag = Config.WEIGHTS[model_name]
    model = timm.create_model(tag, pretrained=True, num_classes=num_classes)
    return model.to(Config.DEVICE)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
        pred = model(images).argmax(1)
        total += labels.size(0)
        correct += pred.eq(labels).sum().item()
    return 100.0 * correct / total


def train_one(model_name: str, optimizer_name: str, lr: float, num_classes: int,
              train_loader, val_loader, test_loader, seed: int, num_epochs: int,
              save_checkpoint: bool = True) -> Dict:
    """Fine-tune one configuration. Epoch selection on val; test evaluated once."""
    Config.set_seed(seed)
    model = build_model(model_name, num_classes)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    gflops = None
    if get_model_complexity_info is not None:
        model_cpu = copy.deepcopy(model).to("cpu").eval()
        macs, _ = get_model_complexity_info(model_cpu, (3, Config.IMG_SIZE, Config.IMG_SIZE),
                                            as_strings=False, print_per_layer_stat=False, verbose=False)
        gflops = macs / 1e9

    print(f"\n{model_name} ({Config.WEIGHTS[model_name]}) + {optimizer_name} lr={lr:g} seed={seed}")
    print(f"  Parameters: {param_count:,}" + (f" | GFLOPs: {gflops:.2f}" if gflops else ""))

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = get_optimizer(model, optimizer_name, lr)
    scheduler = warmup_cosine(optimizer, num_epochs, Config.WARMUP_EPOCHS, len(train_loader))
    use_amp = Config.DEVICE.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val, best_epoch, best_state = -1.0, -1, None
    val_curve = []
    train_time = 0.0

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        model.train()
        for images, labels in tqdm(train_loader, desc=f"epoch {epoch}/{num_epochs}", leave=False):
            images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = criterion(model(images), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        train_time += time.time() - t0

        val_acc = evaluate(model, val_loader)
        val_curve.append(val_acc)
        print(f"  epoch {epoch}: val {val_acc:.2f}%")
        if val_acc > best_val:
            best_val, best_epoch = val_acc, epoch
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    test_acc = evaluate(model, test_loader)
    print(f"  best val {best_val:.2f}% at epoch {best_epoch}; test {test_acc:.2f}%")

    meta = {
        "seed": seed,
        "model": model_name,
        "weights": Config.WEIGHTS[model_name],
        "optimizer": optimizer_name,
        "lr": lr,
        "best_val_acc": best_val,
        "best_epoch": best_epoch,
        "test_acc": test_acc,
        "val_curve": val_curve,
        "param_count": param_count,
        "gflops": gflops,
        "train_time_sec": train_time,
        "num_epochs": num_epochs,
    }
    if save_checkpoint:
        torch.save({"model_state_dict": model.state_dict(), "meta": meta},
                   Config.RESULTS_DIR / f"{model_name}_{optimizer_name}_seed{seed}.pth")
    return meta


def lr_sweep(num_classes, train_loader, val_loader, test_loader) -> Dict[str, Dict[str, float]]:
    """Short sweep, one seed, same grid and budget for both models.

    Returns {model: {optimizer: lr}} and writes it to CHOSEN_LR_FILE.
    """
    seed = Config.SEEDS[0]
    chosen, rows = {}, []
    for model_name in Config.WEIGHTS:
        chosen[model_name] = {}
        for optimizer_name, grid in Config.LR_GRID.items():
            best_lr, best_val = None, -1.0
            for lr in grid:
                meta = train_one(model_name, optimizer_name, lr, num_classes, train_loader, val_loader,
                                 test_loader, seed, Config.SWEEP_EPOCHS, save_checkpoint=False)
                rows.append({"model": model_name, "optimizer": optimizer_name, "lr": lr,
                             "best_val_acc": meta["best_val_acc"]})
                if meta["best_val_acc"] > best_val:
                    best_val, best_lr = meta["best_val_acc"], lr
            chosen[model_name][optimizer_name] = best_lr
            print(f"\nChosen for {model_name} + {optimizer_name}: lr={best_lr:g} (val {best_val:.2f}%)")
    pd.DataFrame(rows).to_csv(Config.RESULTS_DIR / "lr_sweep.csv", index=False)
    Config.CHOSEN_LR_FILE.write_text(json.dumps(chosen, indent=2))
    return chosen


def load_chosen_lrs() -> Dict[str, Dict[str, float]]:
    if Config.CHOSEN_LR_FILE.exists():
        return json.loads(Config.CHOSEN_LR_FILE.read_text())
    print("WARNING: no sweep file found; using the unmatched fallback learning rates. Run --lr-sweep first.")
    return {m: dict(Config.FALLBACK_LR) for m in Config.WEIGHTS}


def report(results_df: pd.DataFrame):
    """Summary table and an honest paired comparison of the best config per model."""
    summary = results_df.groupby(["model", "optimizer"]).agg(
        mean_test=("test_acc", "mean"), std_test=("test_acc", "std"),
        mean_val=("best_val_acc", "mean"), n=("seed", "count"),
        lr=("lr", "first"), weights=("weights", "first"),
    ).sort_values("mean_test", ascending=False).reset_index()
    print("\n" + "=" * 80 + "\nSUMMARY (mean over seeds)\n" + "=" * 80)
    print(summary.to_string(index=False))

    # Best optimizer per model is chosen on validation accuracy, not test.
    best = {}
    for model_name in Config.WEIGHTS:
        sub = summary[summary.model == model_name]
        best[model_name] = sub.loc[sub.mean_val.idxmax(), "optimizer"]

    seeds = sorted(results_df.seed.unique())
    def scores(model_name):
        sub = results_df[(results_df.model == model_name) & (results_df.optimizer == best[model_name])]
        return sub.set_index("seed").loc[seeds, "test_acc"].to_numpy()

    vit, res = scores("ViT-Small"), scores("ResNet-34")
    diff = vit - res
    n = len(diff)
    print("\n" + "=" * 80 + "\nPAIRED COMPARISON (best optimizer per model, chosen on val)\n" + "=" * 80)
    print(f"ViT-Small + {best['ViT-Small']}: {vit.round(2).tolist()}")
    print(f"ResNet-34 + {best['ResNet-34']}: {res.round(2).tolist()}")
    print(f"Gap (ViT minus ResNet): mean {diff.mean():.2f} pp, sd {diff.std(ddof=1):.2f} pp over {n} seeds")
    if n >= 2:
        ci = stats.t.interval(0.95, n - 1, loc=diff.mean(), scale=stats.sem(diff))
        p = stats.ttest_rel(vit, res).pvalue
        print(f"95% CI for the gap: [{ci[0]:.2f}, {ci[1]:.2f}] pp; paired t-test p = {p:.2g}")
    print("Note: the seeds measure optimisation noise around one recipe. The CI is about that noise,")
    print("not about whether the ranking survives a different dataset, budget, or pretraining.")
    if n < 6:
        print(f"Wilcoxon is not reported: with N={n} its smallest possible two-sided p is {2 / 2**n:.4f}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lr-sweep", action="store_true", help="run the learning-rate sweep and exit")
    parser.add_argument("--seeds", type=int, nargs="*", default=None, help="subset of seeds to run")
    parser.add_argument("--epochs", type=int, default=Config.NUM_EPOCHS)
    args = parser.parse_args()

    Config.RESULTS_DIR.mkdir(exist_ok=True)
    seeds = args.seeds or Config.SEEDS
    print(f"PyTorch {torch.__version__} | device {Config.DEVICE}")
    print(f"Weights: {Config.WEIGHTS}")

    if args.lr_sweep:
        loaders = get_dataloaders(seed=Config.SEEDS[0])
        train_loader, val_loader, test_loader, num_classes = loaders
        lr_sweep(num_classes, train_loader, val_loader, test_loader)
        return

    chosen = load_chosen_lrs()
    print(f"Learning rates: {chosen}")
    all_results = []
    for seed in seeds:
        print(f"\n{'=' * 80}\nSeed {seed}\n{'=' * 80}")
        train_loader, val_loader, test_loader, num_classes = get_dataloaders(seed=seed)
        for model_name in Config.WEIGHTS:
            for optimizer_name in Config.LR_GRID:
                meta = train_one(model_name, optimizer_name, chosen[model_name][optimizer_name], num_classes,
                                 train_loader, val_loader, test_loader, seed, args.epochs)
                all_results.append(meta)
                pd.DataFrame(all_results).drop(columns=["val_curve"]).to_csv(
                    Config.RESULTS_DIR / "all_results.csv", index=False)
                (Config.RESULTS_DIR / "all_results.json").write_text(json.dumps(all_results, indent=2))

    report(pd.DataFrame(all_results))
    print(f"\nAll results saved to {Config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
