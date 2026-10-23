"""
ViT-Small vs ResNet-34 fine-tuned on Food-101 under one protocol.

Both models load ImageNet-1k-only weights by explicit timm tag (the timm default
for the ViT is ImageNet-21k), see the same data pipeline, get a learning rate
chosen for each by the same sweep on a validation split, and are evaluated on
the test set once, with the checkpoint that did best on validation. Five seeds.

Usage:
    python comparison.py --lr-sweep                  # pick LR per model and optimizer
    python comparison.py                             # full run with chosen LRs, all in one process
    python comparison.py --run ViT-Small SGD 42      # one configuration (used by the Slurm job array)
    python comparison.py --report                    # merge runs/*.json and print the summary
    python comparison.py --smoke                     # a few batches of everything, to test the setup
"""

import argparse
import copy
import json
import os
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
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

try:
    from ptflops import get_model_complexity_info
except ImportError:
    get_model_complexity_info = None


class Config:
    SEEDS: List[int] = [42, 101, 1337, 2024, 888]
    DEVICE = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    NUM_EPOCHS: int = 10
    WARMUP_EPOCHS: int = 1
    BATCH_SIZE: int = 128  # fits every GPU on the cluster, including 12 GB cards
    IMG_SIZE: int = 224
    NUM_WORKERS: int = 8
    VAL_FRACTION: float = 0.1
    RESULTS_DIR: Path = Path("fair_comparison_results")
    # Where torchvision finds food-101/. Jobs on the cluster point this at node-local disk,
    # because 101,000 small files on NFS are far slower than one tarball copied per job.
    DATA_ROOT: Path = Path(os.environ.get("FOOD101_ROOT", "./data"))
    CHOSEN_LR_FILE: Path = RESULTS_DIR / "chosen_learning_rates.json"

    # ImageNet-1k only, for both.
    WEIGHTS = {
        "ViT-Small": "vit_small_patch16_224.augreg_in1k",
        "ResNet-34": "resnet34.a1_in1k",
    }

    # Sweep grid per optimizer: every value, every model, one seed, short budget.
    LR_GRID = {
        "AdamW": [1e-4, 3e-4, 1e-3],
        "SGD": [1e-2, 3e-2, 1e-1],
    }
    SWEEP_EPOCHS: int = 3
    WEIGHT_DECAY = {"AdamW": 0.05, "SGD": 1e-4}
    MOMENTUM = 0.9

    # Used only when no sweep file exists; not tuned to either model.
    FALLBACK_LR = {"AdamW": 3e-4, "SGD": 1e-2}
    SMOKE: bool = False  # set by --smoke: a handful of batches per loader

    @staticmethod
    def set_seed(seed: int):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)


def autocast():
    """Mixed precision where the hardware has it: fp16 on CUDA, bf16 on Apple GPUs, none on CPU."""
    kind = Config.DEVICE.type
    if kind == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    if kind == "mps":
        return torch.autocast("mps", dtype=torch.bfloat16)
    return torch.autocast("cpu", enabled=False)


def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed + worker_id)


def get_dataloaders(seed: int, batch_size=Config.BATCH_SIZE, img_size=Config.IMG_SIZE):
    """Food-101 with a seeded validation split carved from the training set; the split is
    the same for both models within a seed, and validation uses the test transform."""
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

    train_full = torchvision.datasets.Food101(root=Config.DATA_ROOT, split="train", download=False, transform=train_transform)
    val_full = torchvision.datasets.Food101(root=Config.DATA_ROOT, split="train", download=False, transform=eval_transform)
    test_dataset = torchvision.datasets.Food101(root=Config.DATA_ROOT, split="test", download=False, transform=eval_transform)

    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(train_full))
    n_val = int(len(train_full) * Config.VAL_FRACTION)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    pin = Config.DEVICE.type == "cuda"
    train_loader = DataLoader(Subset(train_full, train_idx), batch_size=batch_size, shuffle=True,
                              num_workers=Config.NUM_WORKERS, pin_memory=pin, worker_init_fn=worker_init_fn,
                              persistent_workers=Config.NUM_WORKERS > 0)
    val_loader = DataLoader(Subset(val_full, val_idx), batch_size=batch_size, shuffle=False,
                            num_workers=Config.NUM_WORKERS, pin_memory=pin, worker_init_fn=worker_init_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=Config.NUM_WORKERS, pin_memory=pin, worker_init_fn=worker_init_fn)

    num_classes = len(train_full.classes)
    print(f"Data: {len(train_idx)} train, {len(val_idx)} val, {len(test_dataset)} test ({num_classes} classes)")
    return train_loader, val_loader, test_loader, num_classes


def is_no_decay_param(name: str, param: torch.Tensor) -> bool:
    """Biases and normalisation parameters get no weight decay. timm names batch norm
    'bn1', 'bn2', 'downsample.1' and layer norm 'norm', 'norm1', 'norm2'."""
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


def take(loader, n_batches=None):
    """Iterate a loader, stopping early in smoke mode."""
    limit = 3 if Config.SMOKE else n_batches
    for i, batch in enumerate(loader):
        if limit is not None and i >= limit:
            break
        yield batch


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
    for images, labels in take(loader):
        images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
        with autocast():
            pred = model(images).argmax(1)
        total += labels.size(0)
        correct += pred.eq(labels).sum().item()
    return 100.0 * correct / total


def train_one(model_name: str, optimizer_name: str, lr: float, num_classes: int,
              train_loader, val_loader, test_loader, seed: int, num_epochs: int,
              save_checkpoint: bool = True) -> Dict:
    """Fine-tune one configuration; pick the epoch on validation, evaluate test once."""
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
    scaler = torch.amp.GradScaler("cuda", enabled=Config.DEVICE.type == "cuda")

    best_val, best_epoch, best_state = -1.0, -1, None
    val_curve = []
    train_time = 0.0

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        model.train()
        for images, labels in tqdm(take(train_loader), desc=f"epoch {epoch}/{num_epochs}", leave=False,
                                   total=3 if Config.SMOKE else len(train_loader), disable=None):
            images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with autocast():
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


def lr_sweep(num_classes, train_loader, val_loader, test_loader, grid_override=None,
             tag="lr_sweep", write_chosen=True) -> Dict[str, Dict[str, float]]:
    """Short sweep, one seed, same grid and budget for both models. Returns
    {model: {optimizer: lr}}; grid_override replaces the grid (used to extend it)."""
    seed = Config.SEEDS[0]
    grids = grid_override or Config.LR_GRID
    chosen, rows = {}, []
    for model_name in Config.WEIGHTS:
        chosen[model_name] = {}
        for optimizer_name, grid in grids.items():
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
    pd.DataFrame(rows).to_csv(Config.RESULTS_DIR / f"{tag}.csv", index=False)
    if write_chosen:
        Config.CHOSEN_LR_FILE.write_text(json.dumps(chosen, indent=2))
    return chosen


def load_chosen_lrs() -> Dict[str, Dict[str, float]]:
    if Config.CHOSEN_LR_FILE.exists():
        return json.loads(Config.CHOSEN_LR_FILE.read_text())
    print("WARNING: no sweep file found; using the unmatched fallback learning rates. Run --lr-sweep first.")
    return {m: dict(Config.FALLBACK_LR) for m in Config.WEIGHTS}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lr-sweep", action="store_true", help="run the learning-rate sweep and exit")
    parser.add_argument("--grid", nargs="*", metavar="OPT=LR[,LR]",
                        help="replace the sweep grid, e.g. AdamW=0.003 SGD=0.3; results go to --tag.csv and "
                             "chosen_learning_rates.json is left alone")
    parser.add_argument("--tag", default="lr_sweep")
    parser.add_argument("--seeds", type=int, nargs="*", default=None, help="subset of seeds to run")
    parser.add_argument("--epochs", type=int, default=Config.NUM_EPOCHS)
    parser.add_argument("--run", nargs=3, metavar=("MODEL", "OPTIMIZER", "SEED"),
                        help="train one configuration only and write runs/<name>.json (for job arrays)")
    parser.add_argument("--report", action="store_true", help="merge runs/*.json and print the summary")
    parser.add_argument("--smoke", action="store_true",
                        help="run a few batches of everything to check the setup; results are meaningless")
    args = parser.parse_args()
    if args.smoke:
        Config.SMOKE = True
        args.epochs = 1
        Config.SWEEP_EPOCHS = 1
        seeds_override = [Config.SEEDS[0]]
    else:
        seeds_override = None

    Config.RESULTS_DIR.mkdir(exist_ok=True)
    seeds = args.seeds or seeds_override or Config.SEEDS
    print(f"PyTorch {torch.__version__} | device {Config.DEVICE}")
    print(f"Weights: {Config.WEIGHTS}")

    if args.lr_sweep:
        train_loader, val_loader, test_loader, num_classes = get_dataloaders(seed=Config.SEEDS[0])
        grid = None
        if args.grid:
            grid = {item.split("=")[0]: [float(v) for v in item.split("=")[1].split(",")] for item in args.grid}
        lr_sweep(num_classes, train_loader, val_loader, test_loader, grid_override=grid,
                 tag=args.tag, write_chosen=grid is None)
        return

    if args.report:
        from report import main as report_main
        report_main()
        return

    chosen = load_chosen_lrs()
    print(f"Learning rates: {chosen}")

    if args.run:
        model_name, optimizer_name, seed = args.run[0], args.run[1], int(args.run[2])
        train_loader, val_loader, test_loader, num_classes = get_dataloaders(seed=seed)
        meta = train_one(model_name, optimizer_name, chosen[model_name][optimizer_name], num_classes,
                         train_loader, val_loader, test_loader, seed, args.epochs)
        runs_dir = Config.RESULTS_DIR / "runs"
        runs_dir.mkdir(exist_ok=True)
        (runs_dir / f"{model_name}_{optimizer_name}_seed{seed}.json").write_text(json.dumps(meta, indent=2))
        return

    all_results = []
    for seed in seeds:
        print(f"\nSeed {seed}")
        train_loader, val_loader, test_loader, num_classes = get_dataloaders(seed=seed)
        for model_name in Config.WEIGHTS:
            for optimizer_name in Config.LR_GRID:
                meta = train_one(model_name, optimizer_name, chosen[model_name][optimizer_name], num_classes,
                                 train_loader, val_loader, test_loader, seed, args.epochs)
                all_results.append(meta)
                pd.DataFrame(all_results).drop(columns=["val_curve"]).to_csv(
                    Config.RESULTS_DIR / "all_results.csv", index=False)
                (Config.RESULTS_DIR / "all_results.json").write_text(json.dumps(all_results, indent=2))

    from report import report
    report(pd.DataFrame(all_results))
    print(f"\nAll results saved to {Config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
