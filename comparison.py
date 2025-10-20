"""
Fair Architectural Comparison: ViT-Small vs ResNet-34
Author: Ambar
Institution: National University of Singapore

Methodology:
- Identical augmentation for both architectures
- Parameter-matched models (~22M parameters)
- Multiple random seeds (N=5) with statistical testing
- Optimizer ablation study (AdamW vs SGD)
- Full reproducibility with deterministic training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm.notebook import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import timm
import time
import warnings
import copy
from pathlib import Path
from scipy.stats import ttest_rel, wilcoxon
import pandas as pd
from typing import List, Dict
from ptflops import get_model_complexity_info

# Configuration
class Config:
    SEEDS: List[int] = [42, 101, 1337, 2024, 888]
    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_EPOCHS: int = 10
    BATCH_SIZE: int = 512 # Adjust based on your GPU memory
    IMG_SIZE: int = 224
    NUM_WORKERS: int = 8 # Adjust based on your GPU cores
    RESULTS_DIR: Path = Path("fair_comparison_results")
    
    OPTIMIZERS = {
        'AdamW': {'lr': 3e-4, 'weight_decay': 0.05},
        'SGD': {'lr': 1e-2, 'weight_decay': 1e-4, 'momentum': 0.9}
    }
    
    @staticmethod
    def set_seed(seed: int):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

warnings.filterwarnings('ignore')
Config.RESULTS_DIR.mkdir(exist_ok=True)

print(f"PyTorch: {torch.__version__} | Device: {Config.DEVICE}")
print(f"Experiment: {len(Config.SEEDS)} seeds for statistical robustness\n")

# Data Pipeline
def worker_init_fn(worker_id):
    """Seed dataloader workers for reproducibility"""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed + worker_id)

def get_dataloaders(batch_size=Config.BATCH_SIZE, img_size=Config.IMG_SIZE):
    """Load Food-101 with identical augmentation for both architectures"""
    
    train_transform = transforms.Compose([
        transforms.TrivialAugmentWide(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_dataset = torchvision.datasets.Food101(
        root='./data', split='train', download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.Food101(
        root='./data', split='test', download=True, transform=test_transform
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=Config.NUM_WORKERS, pin_memory=True, worker_init_fn=worker_init_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=Config.NUM_WORKERS, pin_memory=True, worker_init_fn=worker_init_fn
    )
    
    num_classes = len(train_dataset.classes)
    print(f"Data: {len(train_dataset)} train, {len(test_dataset)} test ({num_classes} classes)")
    return train_loader, test_loader, num_classes

# Optimizer Configuration
def get_optimizer(model, optimizer_name: str, hps: Dict):
    """Separate parameters for proper weight decay"""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or 'norm' in name:
            no_decay.append(param)
        else:
            decay.append(param)
    
    params = [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': hps['weight_decay']}
    ]
    
    if optimizer_name == 'AdamW':
        return optim.AdamW(params, lr=hps['lr'])
    elif optimizer_name == 'SGD':
        return optim.SGD(params, lr=hps['lr'], momentum=hps['momentum'])
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

# Training Pipeline
def fine_tune_model(model_name: str, optimizer_name: str, num_classes: int,
                   train_loader: DataLoader, test_loader: DataLoader, seed: int) -> Dict:
    """Train model with specified configuration"""
    
    Config.set_seed(seed)
    
    # Initialize model
    if model_name == 'ViT-Small':
        model = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=num_classes)
    elif model_name == 'ResNet-34':
        model = timm.create_model('resnet34', pretrained=True, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model = model.to(Config.DEVICE)
    
    # Log model statistics
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_cpu = copy.deepcopy(model).to('cpu').eval()
    macs, _ = get_model_complexity_info(
        model_cpu, (3, Config.IMG_SIZE, Config.IMG_SIZE),
        as_strings=False, print_per_layer_stat=False, verbose=False
    )
    
    print(f"\n{model_name} + {optimizer_name} (Seed {seed}):")
    print(f"  Parameters: {param_count:,} | GFLOPs: {macs / 1e9:.2f}")
    
    # Training setup
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = get_optimizer(model, optimizer_name, Config.OPTIMIZERS[optimizer_name])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS)
    scaler = torch.cuda.amp.GradScaler(enabled=(Config.DEVICE.type == 'cuda'))
    
    best_acc = 0.0
    best_model_state = None
    total_train_time = 0
    
    # Training loop
    for epoch in range(1, Config.NUM_EPOCHS + 1):
        epoch_start = time.time()
        
        # Train
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
            optimizer.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast(enabled=(Config.DEVICE.type == 'cuda')):
                logits = model(images)
                loss = criterion(logits, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        scheduler.step()
        total_train_time += time.time() - epoch_start
        
        # Evaluate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
                logits = model(images)
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        acc = 100. * correct / total
        if acc > best_acc:
            best_acc = acc
            best_model_state = copy.deepcopy(model.state_dict())
    
    # Save best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    run_meta = {
        'seed': seed,
        'model': model_name,
        'optimizer': optimizer_name,
        'best_acc': best_acc,
        'param_count': param_count,
        'gflops': macs / 1e9,
        'train_time_sec': total_train_time
    }
    
    torch.save(
        {'model_state_dict': model.state_dict(), 'meta': run_meta},
        Config.RESULTS_DIR / f"{model_name}_{optimizer_name}_seed{seed}.pth"
    )
    
    print(f"  Best accuracy: {best_acc:.2f}%")
    return run_meta

# Main Execution
if __name__ == '__main__':
    train_loader, test_loader, num_classes = get_dataloaders()
    
    model_names = ['ViT-Small', 'ResNet-34']
    optimizer_names = ['AdamW', 'SGD']
    all_results = []
    
    # Run all experiments
    for seed in Config.SEEDS:
        print(f"\n{'='*80}\nSeed: {seed}\n{'='*80}")
        for model_name in model_names:
            for optimizer_name in optimizer_names:
                run_meta = fine_tune_model(
                    model_name, optimizer_name, num_classes,
                    train_loader, test_loader, seed
                )
                all_results.append(run_meta)
    
    # Aggregate results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(Config.RESULTS_DIR / 'all_results.csv', index=False)
    
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    summary_df = results_df.groupby(['model', 'optimizer']).agg(
        mean_accuracy=('best_acc', 'mean'),
        std_accuracy=('best_acc', 'std'),
        mean_time=('train_time_sec', 'mean'),
        gflops=('gflops', 'first')
    ).sort_values('mean_accuracy', ascending=False).reset_index()
    
    print("\n" + summary_df.to_string())
    
    # Statistical Analysis
    print("\n" + "="*80)
    print("STATISTICAL SIGNIFICANCE TESTING")
    print("="*80)
    
    mean_by_cfg = results_df.groupby(['model', 'optimizer'])['best_acc'].mean().reset_index()
    best_opts = mean_by_cfg.loc[
        mean_by_cfg.groupby('model')['best_acc'].idxmax()
    ].set_index('model')['optimizer'].to_dict()
    
    vit_scores = []
    resnet_scores = []
    for seed in Config.SEEDS:
        vit_val = results_df.loc[
            (results_df['model'] == 'ViT-Small') &
            (results_df['optimizer'] == best_opts['ViT-Small']) &
            (results_df['seed'] == seed), 'best_acc'
        ].values
        
        res_val = results_df.loc[
            (results_df['model'] == 'ResNet-34') &
            (results_df['optimizer'] == best_opts['ResNet-34']) &
            (results_df['seed'] == seed), 'best_acc'
        ].values
        
        assert vit_val.size == 1 and res_val.size == 1, "Missing run"
        vit_scores.append(vit_val.item())
        resnet_scores.append(res_val.item())
    
    # Statistical tests
    ttest_pvalue = ttest_rel(vit_scores, resnet_scores).pvalue
    wilcoxon_pvalue = wilcoxon(vit_scores, resnet_scores).pvalue
    
    def cohens_d(a, b):
        diff = np.array(a) - np.array(b)
        return diff.mean() / diff.std(ddof=1) if diff.std() > 0 else 0
    
    effect_size = cohens_d(vit_scores, resnet_scores)
    
    print(f"\nBest configurations:")
    print(f"  ViT-Small with {best_opts['ViT-Small']}")
    print(f"  ResNet-34 with {best_opts['ResNet-34']}")
    print(f"\nScores across {len(Config.SEEDS)} seeds:")
    print(f"  ViT:    {[f'{s:.2f}' for s in vit_scores]}")
    print(f"  ResNet: {[f'{s:.2f}' for s in resnet_scores]}")
    print(f"\nStatistical tests:")
    print(f"  Paired t-test p-value: {ttest_pvalue:.4f}")
    print(f"  Wilcoxon p-value:      {wilcoxon_pvalue:.4f}")
    print(f"  Cohen's d:             {effect_size:.3f}")
    
    # Conclusion
    winner = "ViT-Small" if np.mean(vit_scores) > np.mean(resnet_scores) else "ResNet-34"
    if ttest_pvalue < 0.05:
        print(f"\nConclusion: {winner} significantly outperforms (p < 0.05)")
    else:
        print(f"\nConclusion: No significant difference (p ≥ 0.05)")
    
    print("\n" + "="*80)
    print("All results saved to:", Config.RESULTS_DIR)
    print("="*80)
