# ==============================================================================
# Analysis can be carried out after training is complete. Below is the suggested code. This was done in parts while conducting the analysis.
# ==============================================================================


# Important Note:
# models/ contains fine tuned models saved (Currently this has saved models for seed 42 and seed 101). These are just for reference and may be deleted before starting the experiment
   

# ==============================================================================
# Step 1: Import necessary libraries
# ==============================================================================

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import timm
from pathlib import Path
import pandas as pd
from collections import defaultdict
import copy
import warnings
from tqdm import tqdm # Use standard tqdm
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, accuracy_score
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
from scipy.stats import spearmanr, ttest_rel, wilcoxon

# Ignore common warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# 2. CONFIGURATION
# ==============================================================================
class Config:
    # --- Experiment Setup (Should match training script values) ---
    SEEDS_TRAINED: list[int] = [42, 101, 1337, 2024, 888] # List of seeds used during training
    NUM_CLASSES: int = 101 # For Food-101 dataset
    IMG_SIZE: int = 224

    # --- Analysis Setup ---
    ANALYSIS_SEED: int = 42 # Seed for reproducible analysis steps (t-SNE, subsetting)
    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE: int = 512 # Batch size for feature extraction (adjust based on GPU memory)
    NUM_WORKERS: int = 8   # Dataloader workers (adjust based on system; 2 might be safer on Colab)
    RESULTS_DIR: Path = Path("fair-vit-resnet-comparison") #Enter path
    RESULTS_CSV: Path = RESULTS_DIR / "fair_comparison_results.csv" #Enter path
    RDM_SUBSET_SIZE: int = 500 # Number of images for RDM calculation & plotting
    TSNE_SUBSET_SIZE: int = 1000 # Number of images for t-SNE plot
    CM_SUBSET_CLASSES: int = 20 # Max classes for confusion matrix plot clarity
    CALIBRATION_BINS: int = 15

    # --- Plotting ---
    PLOT_STYLE: str = 'seaborn-v0_8-whitegrid'
    COLORS = {'ViT-Small_AdamW': '#2E86AB', 'ViT-Small_SGD': '#45B7D1',
              'ResNet-34_AdamW': '#A23B72', 'ResNet-34_SGD': '#FF6B6B'}
    LINESTYLES = {'ViT-Small_AdamW': '-', 'ViT-Small_SGD': '-.',
                  'ResNet-34_AdamW': '--', 'ResNet-34_SGD': ':'}

print(f"{'='*80}\n🔬 Consolidated Analysis Suite\n{'='*80}")
print(f"PyTorch Version: {torch.__version__}")
print(f"Device: {Config.DEVICE}")
print(f"Analysis Seed: {Config.ANALYSIS_SEED}")
print(f"Results Directory: {Config.RESULTS_DIR}")
if not Config.RESULTS_CSV.exists():
    raise FileNotFoundError(f"Results CSV not found at {Config.RESULTS_CSV}. Please ensure training script ran and generated the file.")
if not any(Config.RESULTS_DIR.glob("*.pth")):
    warnings.warn(f"No .pth model checkpoints found in {Config.RESULTS_DIR}. Detailed analysis requiring model loading will fail.")


# Set seed for analysis reproducibility
np.random.seed(Config.ANALYSIS_SEED)
torch.manual_seed(Config.ANALYSIS_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(Config.ANALYSIS_SEED)
    # Note: Setting deterministic operations can impact performance
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

# ==============================================================================
# 3. HELPER FUNCTIONS (Data Loading, Model Handling, Calculations)
# ==============================================================================

def get_dataloader(batch_size=Config.BATCH_SIZE, img_size=Config.IMG_SIZE, num_workers=Config.NUM_WORKERS):
    """Loads the Food-101 test dataset."""
    print(f"\n📦 Loading Food-101 test data (Size: {img_size}x{img_size})...")
    # Standard ImageNet normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # Common test transform: resize slightly larger, center crop
    test_transform = transforms.Compose([
        transforms.Resize(img_size + 32, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize,
    ])

    try:
        # Assumes dataset is already downloaded to ./data directory
        test_dataset = torchvision.datasets.Food101(root='./data', split='test', download=False, transform=test_transform)
    except RuntimeError as e:
         print(f"🛑 Error loading dataset: {e}")
         print("   Ensure the Food-101 dataset is downloaded and available in the './data' directory.")
         raise

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True if Config.DEVICE.type == 'cuda' else False)
    print(f"✓ Test data ready: {len(test_dataset)} samples, {len(test_dataset.classes)} classes.")
    return test_loader, test_dataset.classes

def get_model(model_name_base: str, num_classes: int = Config.NUM_CLASSES) -> nn.Module:
    """Instantiates a model architecture without pretrained weights."""
    print(f"   Instantiating {model_name_base}...")
    if 'ViT-Small' in model_name_base:
        model = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=num_classes)
    elif 'ResNet-34' in model_name_base:
        model = timm.create_model('resnet34', pretrained=False, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model name base: {model_name_base}")
    return model.to(Config.DEVICE)

def load_checkpoint(model: nn.Module, checkpoint_path: Path):
    """Loads state dict from a checkpoint file into the model."""
    print(f"   Loading weights from: {checkpoint_path.name}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE)
        # Handle checkpoints saved with 'model_state_dict' key or just the state_dict
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        # Load weights, ignoring unexpected keys and reporting missing ones
        load_result = model.load_state_dict(state_dict, strict=False)
        if load_result.missing_keys:
            print(f"   ⚠️ Warning: Missing keys found during loading: {load_result.missing_keys}")
        if load_result.unexpected_keys:
            print(f"   ⚠️ Warning: Unexpected keys found during loading: {load_result.unexpected_keys}")
    except Exception as e:
        print(f"   🛑 Error loading checkpoint {checkpoint_path.name}: {e}")
        raise
    return model

def _get_feature_extractor(model: nn.Module) -> nn.Module:
    """Returns the feature extraction part of the model (before the final classifier)."""
    if isinstance(model, timm.models.VisionTransformer):
        feature_extractor = copy.deepcopy(model)
        feature_extractor.head = nn.Identity() # Remove the classification head
        return feature_extractor
    elif isinstance(model, timm.models.ResNet):
        # Return model up to the layer before the final fully connected layer
        return nn.Sequential(*list(model.children())[:-1])
    else:
        raise TypeError(f"Feature extraction not implemented for model type: {type(model)}")

@torch.no_grad()
def get_predictions_and_features(model: nn.Module, dataloader: DataLoader, model_id: str) -> dict:
    """Extracts predictions, probabilities, and features for all samples in dataloader."""
    model.eval()
    feature_extractor = _get_feature_extractor(model).to(Config.DEVICE)
    all_preds, all_labels, all_probs, all_features = [], [], [], []
    use_amp = (Config.DEVICE.type == 'cuda') # Enable mixed precision for potential speedup

    for images, labels in tqdm(dataloader, desc=f"Extracting outputs for {model_id}"):
        images = images.to(Config.DEVICE)
        current_labels = labels.numpy() # Store labels before moving images to device

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images)
            # Ensure features are flattened to (batch_size, feature_dim)
            features = feature_extractor(images).flatten(start_dim=1)

        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_labels.append(current_labels)
        all_probs.append(torch.softmax(logits, dim=1).float().cpu().numpy()) # Use float() for safety
        all_features.append(features.float().cpu().numpy())

    # Concatenate results from all batches
    return {
        "predictions": np.concatenate(all_preds),
        "labels": np.concatenate(all_labels),
        "probabilities": np.concatenate(all_probs),
        "features": np.concatenate(all_features)
    }

def compute_rdm(features: np.ndarray, name: str) -> np.ndarray:
    """Computes the Representational Dissimilarity Matrix (1 - Spearman correlation)."""
    print(f"Computing RDM for {name} ({features.shape[0]} feature vectors)...")
    # Using Spearman's rank correlation is common for RDMs
    correlation_matrix, p_matrix = spearmanr(features, axis=1)
    rdm = 1 - correlation_matrix
    np.fill_diagonal(rdm, 0) # Distance to self is 0
    return rdm

def compare_rdms(rdm1: np.ndarray, rdm2: np.ndarray) -> tuple[float, float]:
    """Compares two RDMs using Spearman correlation on the upper triangle elements."""
    if rdm1.shape != rdm2.shape:
        raise ValueError("RDMs must have the same shape for comparison.")
    # Extract upper triangle elements (excluding the diagonal)
    indices = np.triu_indices(rdm1.shape[0], k=1)
    rdm1_flat = rdm1[indices]
    rdm2_flat = rdm2[indices]
    correlation, p_value = spearmanr(rdm1_flat, rdm2_flat)
    print(f"   Spearman correlation between RDMs: {correlation:.4f} (p={p_value:.3g})")
    return correlation, p_value

def calculate_calibration_data(probs, labels, n_bins=Config.CALIBRATION_BINS):
    """Calculates Expected Calibration Error (ECE) and data for reliability diagrams."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)

    ece = 0.0
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    bin_accuracies = np.zeros(n_bins)
    bin_confidences = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    for i in range(n_bins):
        # Find samples where confidence falls into the current bin
        in_bin = (confidences > bin_lowers[i]) & (confidences <= bin_uppers[i])
        bin_counts[i] = np.sum(in_bin)

        if bin_counts[i] > 0:
            # Calculate mean accuracy and confidence for samples in the bin
            accuracy_in_bin = np.mean(accuracies[in_bin])
            confidence_in_bin = np.mean(confidences[in_bin])
            bin_accuracies[i] = accuracy_in_bin
            bin_confidences[i] = confidence_in_bin
            # Add weighted difference to ECE
            ece += np.abs(accuracy_in_bin - confidence_in_bin) * (bin_counts[i] / len(labels))

    return {
        "ece": ece,
        "bin_accuracies": bin_accuracies,    # Mean accuracy per bin
        "bin_confidences": bin_confidences,  # Mean confidence per bin
        "bin_counts": bin_counts,          # Number of samples per bin
        "bin_lowers": bin_lowers           # For plotting bar widths
    }

def robust_cohens_d(a: list, b: list) -> float:
    """Calculates Cohen's d for paired samples, handling zero standard deviation."""
    diff = np.array(a) - np.array(b)
    n = len(diff)
    if n < 2: return np.nan # Cannot compute std dev with less than 2 points

    mean_diff = np.mean(diff)
    std_dev_diff = np.std(diff, ddof=1) # Use sample standard deviation (ddof=1)

    if std_dev_diff == 0:
        # If std dev is 0, the difference is perfectly consistent across all pairs.
        print(f"   Note: Standard deviation of differences is 0. Mean difference = {mean_diff:.4f}")
        # Cohen's d is technically infinite if mean_diff != 0, or 0 if mean_diff == 0.
        # Returning np.inf for non-zero difference is mathematically correct but may need special handling in reports.
        return np.inf if mean_diff != 0 else 0.0
    else:
        return mean_diff / std_dev_diff

# ==============================================================================
# 5. MAIN ANALYSIS EXECUTION SCRIPT
# ==============================================================================
if __name__ == '__main__':
    plt.style.use(Config.PLOT_STYLE)

    # --- A. Load Overall Results from CSV ---
    print(f"\n📈 Loading overall results from {Config.RESULTS_CSV}...")
    try:
        results_df = pd.read_csv(Config.RESULTS_CSV)
    except FileNotFoundError:
        print(f"🛑 Error: Results CSV file not found at {Config.RESULTS_CSV}.")
        exit() # Cannot proceed without the summary CSV

    print("✓ Overall results loaded:")
    # Display the dataframe cleanly
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
       print(results_df)

    # Calculate Summary Statistics (Mean ± Std over seeds)
    summary_df = results_df.groupby(['model', 'optimizer']).agg(
        mean_accuracy=('best_acc', 'mean'),
        std_accuracy=('best_acc', 'std'),
        mean_time=('train_time_sec', 'mean'),
        gflops=('gflops', 'first'),
        param_count=('param_count', 'first')
    ).sort_values('mean_accuracy', ascending=False).reset_index()

    print("\n" + "="*80 + "\n📊 SUMMARY (Mean ± Std over available seeds)\n" + "="*80)
    # Format for better readability
    summary_df['mean_accuracy_str'] = summary_df.apply(lambda row: f"{row['mean_accuracy']:.2f}% ± {row['std_accuracy']:.2f}", axis=1)
    print(summary_df[['model', 'optimizer', 'mean_accuracy_str', 'mean_time', 'gflops', 'param_count']].to_string(index=False))


    # --- B. Perform Statistical Significance Testing (using CSV data) ---
    print("\n" + "="*80 + "\n⚖️ STATISTICAL SIGNIFICANCE & EFFECT SIZE (Based on CSV)\n" + "="*80)

    # Determine the best optimizer for each base model architecture based on mean accuracy
    best_opts_idx = summary_df.loc[summary_df.groupby('model')['mean_accuracy'].idxmax()]
    best_vit_config = best_opts_idx[best_opts_idx['model'] == 'ViT-Small']
    best_resnet_config = best_opts_idx[best_opts_idx['model'] == 'ResNet-34']

    if best_vit_config.empty or best_resnet_config.empty:
        print("🛑 Error: Could not determine best ViT and ResNet configurations from summary data.")
        exit()

    best_vit_opt = best_vit_config['optimizer'].iloc[0]
    best_resnet_opt = best_resnet_config['optimizer'].iloc[0]
    print(f"   Best ViT config found: ViT-Small + {best_vit_opt}")
    print(f"   Best ResNet config found: ResNet-34 + {best_resnet_opt}")

    # Extract accuracy scores for these best configurations across all available seeds
    vit_scores = results_df[(results_df['model'] == 'ViT-Small') & (results_df['optimizer'] == best_vit_opt)]['best_acc'].tolist()
    resnet_scores = results_df[(results_df['model'] == 'ResNet-34') & (results_df['optimizer'] == best_resnet_opt)]['best_acc'].tolist()
    num_seeds_found = len(vit_scores) # Assuming paired data

    # Perform tests only if we have paired scores for at least 2 seeds
    if num_seeds_found >= 2 and len(vit_scores) == len(resnet_scores):
        print(f"\nComparing best ViT ({best_vit_opt}) vs. best ResNet ({best_resnet_opt}) across {num_seeds_found} seeds:")
        print(f"  - ViT Scores: {[f'{s:.2f}' for s in vit_scores]}")
        print(f"  - ResNet Scores: {[f'{s:.2f}' for s in resnet_scores]}")

        # Paired t-test
        ttest_stat, ttest_pvalue = ttest_rel(vit_scores, resnet_scores)
        print(f"  - Paired t-test: statistic={ttest_stat:.3f}, p-value={ttest_pvalue:.4f}")

        # Wilcoxon signed-rank test
        try:
            # Need at least a few pairs, sometimes fails with small N or identical differences
            wilcoxon_stat, wilcoxon_pvalue = wilcoxon(vit_scores, resnet_scores)
            print(f"  - Wilcoxon test: statistic={wilcoxon_stat:.1f}, p-value={wilcoxon_pvalue:.4f}")
        except ValueError as e:
            print(f"  - Wilcoxon test: Could not be computed ({e}). Requires N > ~5-10 or non-identical differences.")
            wilcoxon_pvalue = np.nan # Use NaN to indicate it wasn't computed

        # Effect Size (Cohen's d for paired samples)
        effect_size = robust_cohens_d(vit_scores, resnet_scores)
        if np.isinf(effect_size):
             print(f"  - Cohen's d (Effect Size): Not finite (indicates perfectly consistent difference)")
        else:
             print(f"  - Cohen's d (Effect Size): {effect_size:.3f}")

        # Final Conclusion based on p-value < 0.05 (primarily t-test)
        alpha = 0.05
        if ttest_pvalue < alpha:
            winner = "ViT-Small" if np.mean(vit_scores) > np.mean(resnet_scores) else "ResNet-34"
            print(f"\n  ✅ Conclusion: The difference is STATISTICALLY SIGNIFICANT (p < {alpha}). {winner} performed significantly better.")
        else:
            print(f"\n  ❌ Conclusion: The difference is NOT statistically significant (p >= {alpha}). Performance is comparable.")
            # Note: Wilcoxon p-value might differ, but t-test is standard for paired, likely normal data.
            if not np.isnan(wilcoxon_pvalue) and wilcoxon_pvalue < alpha:
                print(f"     (Note: Wilcoxon test suggested significance p={wilcoxon_pvalue:.4f}, but t-test is primary here).")

    else:
        print(f"🛑 Error: Could not perform statistical tests. Need paired results for >= 2 seeds.")
        print(f"   Found {len(vit_scores)} ViT scores and {len(resnet_scores)} ResNet scores.")


    # --- C. Load Models and Data for Detailed Analysis ---
    # Load test dataloader and class names
    try:
        test_loader, class_names = get_dataloader()
    except Exception as e:
        print(f"🛑 Error initializing dataloader: {e}. Cannot proceed with detailed analysis.")
        exit()

    detailed_analysis_data = {} # Store features, predictions etc.
    representative_models = {} # Store loaded model objects

    # Identify the specific seed runs corresponding to the best accuracy for each best configuration
    # (Or just use a fixed seed like Config.ANALYSIS_SEED if preferred for consistency)
    best_vit_run_details = results_df.loc[results_df[(results_df['model'] == 'ViT-Small') & (results_df['optimizer'] == best_vit_opt)]['best_acc'].idxmax()]
    best_resnet_run_details = results_df.loc[results_df[(results_df['model'] == 'ResNet-34') & (results_df['optimizer'] == best_resnet_opt)]['best_acc'].idxmax()]

    # Optionally, also analyze the failed ResNet-SGD case from one of the seeds
    failed_resnet_run_details = results_df[(results_df['model'] == 'ResNet-34') & (results_df['optimizer'] == 'SGD')].iloc[0] # Just take the first SGD seed

    # Define which models to load for detailed plots and reports
    models_to_analyze = {
        f"ViT-Small_{best_vit_opt}_seed{int(best_vit_run_details['seed'])}": best_vit_run_details,
        f"ResNet-34_{best_resnet_opt}_seed{int(best_resnet_run_details['seed'])}": best_resnet_run_details,
        f"ResNet-34_SGD_seed{int(failed_resnet_run_details['seed'])}": failed_resnet_run_details, # Add the failed run
    }

    print(f"\n📦 Loading specific model checkpoints for detailed analysis...")
    loaded_model_ids = []
    for model_id, run_details in models_to_analyze.items():
        seed = int(run_details['seed'])
        model_base = run_details['model']
        optimizer = run_details['optimizer']
        checkpoint_fname = f"{model_base}_{optimizer}_seed{seed}.pth"
        checkpoint_path = Config.RESULTS_DIR / checkpoint_fname

        if checkpoint_path.exists():
            try:
                model = get_model(model_base)
                model = load_checkpoint(model, checkpoint_path)
                representative_models[model_id] = model
                loaded_model_ids.append(model_id)
                print(f"✓ Loaded: {model_id}")
            except Exception as e:
                print(f"⚠️ Warning: Could not load checkpoint for {model_id}: {e}")
        else:
            print(f"⚠️ Warning: Checkpoint file not found for {model_id} at {checkpoint_path}. Skipping.")

    # --- D. Extract Features and Predictions for Loaded Models ---
    print("\n⚙️ Extracting features and predictions for detailed analysis...")
    if not loaded_model_ids:
        print("🛑 No models were successfully loaded. Cannot extract features or perform detailed analysis.")
        exit()

    for model_id in loaded_model_ids:
        model = representative_models[model_id]
        try:
            detailed_analysis_data[model_id] = get_predictions_and_features(model, test_loader, model_id)
            print(f"✓ Features extracted for: {model_id}")
        except Exception as e:
            print(f"🛑 Error extracting features for {model_id}: {e}")
            # Remove model if extraction fails
            if model_id in representative_models: del representative_models[model_id]
            loaded_model_ids.remove(model_id)

    # Refresh the list of models we actually have data for
    valid_model_ids = list(detailed_analysis_data.keys())
    if not valid_model_ids:
        print("🛑 Feature extraction failed for all loaded models. Cannot perform detailed analysis.")
        exit()

    # --- E. Perform Detailed Analyses (RDMs, Calibration etc.) ---
    print("\n--- Performing detailed analyses on representative models ---")
    rdms = {}
    rdm_correlations = {}

    # Select a random subset of images for RDM/tSNE (consistent across models)
    num_test_samples = len(detailed_analysis_data[valid_model_ids[0]]['features']) # Use first valid model
    rdm_subset_indices = np.random.choice(num_test_samples, min(Config.RDM_SUBSET_SIZE, num_test_samples), replace=False)
    tsne_subset_indices = np.random.choice(num_test_samples, min(Config.TSNE_SUBSET_SIZE, num_test_samples), replace=False)


    # Compute RDMs
    for model_id in valid_model_ids:
        features_subset = detailed_analysis_data[model_id]['features'][rdm_subset_indices]
        rdms[model_id] = compute_rdm(features_subset, model_id)

    # Compare RDMs (if at least 2 were computed)
    if len(rdms) >= 2:
        print("\nComparing RDMs...")
        # Compare best ViT vs best ResNet RDMs if available
        best_vit_id_loaded = f"ViT-Small_{best_vit_opt}_seed{int(best_vit_run_details['seed'])}"
        best_res_id_loaded = f"ResNet-34_{best_resnet_opt}_seed{int(best_resnet_run_details['seed'])}"

        if best_vit_id_loaded in rdms and best_res_id_loaded in rdms:
            corr, pval = compare_rdms(rdms[best_vit_id_loaded], rdms[best_res_id_loaded])
            rdm_correlations[(best_vit_id_loaded, best_res_id_loaded)] = corr
        else:
             print("   Cannot compare best ViT vs best ResNet RDMs - one or both were not loaded/processed.")

    # Calculate Calibration Data
    calibration_data = {}
    for model_id in valid_model_ids:
        calibration_data[model_id] = calculate_calibration_data(detailed_analysis_data[model_id]['probabilities'],
                                                               detailed_analysis_data[model_id]['labels'])
        print(f"✓ Calibration calculated for {model_id} (ECE={calibration_data[model_id]['ece']:.4f})")

    # ==============================================================================
    # 6. GENERATE VISUALIZATIONS
    # ==============================================================================
    print("\n" + "="*80 + "\n📊 GENERATING PUBLICATION-QUALITY VISUALIZATIONS\n" + "="*80)

    # --- Plot 1: Overall Performance Bar Plot (from CSV) ---
    print("   Generating Plot 1: Overall Performance Summary...")
    try:
        fig1, ax1 = plt.subplots(figsize=(12, 7))
        summary_df_plot = summary_df.sort_values('mean_accuracy', ascending=False) # Ensure order for plotting
        summary_df_plot['config'] = summary_df_plot['model'] + ' + ' + summary_df_plot['optimizer']
        bars = ax1.barh(summary_df_plot['config'], summary_df_plot['mean_accuracy'],
                       xerr=summary_df_plot['std_accuracy'], capsize=5,
                       color=[Config.COLORS.get(cfg, '#CCCCCC') for cfg in summary_df_plot['config']]) # Use config for color lookup
        ax1.set_title(f'Overall Model Performance (Mean ± Std over Seeds)', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Mean Accuracy (%)', fontsize=12)
        ax1.set_ylabel('Configuration', fontsize=12)
        ax1.tick_params(axis='both', which='major', labelsize=10)
        ax1.set_xlim(left=0) # Ensure x-axis starts at 0
        ax1.invert_yaxis() # Show best performing model at the top
        plt.tight_layout()
        plt.savefig(Config.RESULTS_DIR / "P1_overall_performance.png", dpi=300)
        print("✓ Plot 1 saved.")
        plt.show()
    except Exception as e:
        print(f"   ⚠️ Failed to generate Plot 1: {e}")

    # --- Plot 2: Comparative ROC Curves (using loaded representative models) ---
    print(f"\n   Generating Plot 2: Comparative ROC Curves...")
    try:
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        roc_models_plotted = 0
        for model_id in valid_model_ids: # Iterate through models with extracted data
            data = detailed_analysis_data[model_id]
            config_name = '_'.join(model_id.split('_')[:2]) # e.g., ViT-Small_SGD

            y_bin = label_binarize(data['labels'], classes=range(Config.NUM_CLASSES))
            fpr, tpr = dict(), dict()

            # Compute Macro-average ROC
            all_fpr_macro = np.unique(np.concatenate([roc_curve(y_bin[:, i], data['probabilities'][:, i])[0] for i in range(Config.NUM_CLASSES)]))
            mean_tpr_macro = np.zeros_like(all_fpr_macro)
            for i in range(Config.NUM_CLASSES):
                fpr_i, tpr_i, _ = roc_curve(y_bin[:, i], data['probabilities'][:, i])
                mean_tpr_macro += np.interp(all_fpr_macro, fpr_i, tpr_i)
            mean_tpr_macro /= Config.NUM_CLASSES
            roc_auc_macro = auc(all_fpr_macro, mean_tpr_macro)

            ax2.plot(all_fpr_macro, mean_tpr_macro,
                     label=f"{model_id} (Macro AUC = {roc_auc_macro:.3f})", # Use full ID in label
                     color=Config.COLORS.get(config_name, '#CCCCCC'),
                     linestyle=Config.LINESTYLES.get(config_name, '-'), lw=2.5)
            roc_models_plotted += 1

        if roc_models_plotted > 0:
            ax2.plot([0, 1], [0, 1], 'k--', label='Random Chance (AUC = 0.500)')
            ax2.set_xlabel("False Positive Rate", fontsize=12)
            ax2.set_ylabel("True Positive Rate", fontsize=12)
            ax2.set_title("Macro-Average ROC Curves (Representative Models)", fontsize=16, fontweight='bold')
            ax2.legend(fontsize=9) # Smaller font for potentially many labels
            ax2.tick_params(axis='both', which='major', labelsize=10)
            ax2.grid(True)
            plt.tight_layout()
            plt.savefig(Config.RESULTS_DIR / "P2_roc_curves.png", dpi=300)
            print("✓ Plot 2 saved.")
            plt.show()
        else:
            print("   ⚠️ Skipping ROC plot - No valid model data available.")
    except Exception as e:
        print(f"   ⚠️ Failed to generate Plot 2: {e}")


    # --- Plot 3: RDM Comparison (if RDMs computed) ---
    print(f"\n   Generating Plot 3: RDM Comparison...")
    try:
        # Check if we have the specific RDMs for best ViT and best ResNet
        best_vit_id_loaded = f"ViT-Small_{best_vit_opt}_seed{int(best_vit_run_details['seed'])}"
        best_res_id_loaded = f"ResNet-34_{best_resnet_opt}_seed{int(best_resnet_run_details['seed'])}"
        if best_vit_id_loaded in rdms and best_res_id_loaded in rdms:
            fig3, axes3 = plt.subplots(1, 2, figsize=(12, 6))
            fig3.suptitle(f"Representational Dissimilarity Matrices ({Config.RDM_SUBSET_SIZE} Images)", fontsize=16, fontweight='bold')

            # Plot ViT RDM
            im1 = axes3[0].imshow(rdms[best_vit_id_loaded], cmap='viridis', aspect='auto')
            axes3[0].set_title(best_vit_id_loaded, fontsize=12)
            axes3[0].set_xticks([]); axes3[0].set_yticks([])
            fig3.colorbar(im1, ax=axes3[0], fraction=0.046, pad=0.04)

            # Plot ResNet RDM
            im2 = axes3[1].imshow(rdms[best_res_id_loaded], cmap='viridis', aspect='auto')
            rdm_corr_val = rdm_correlations.get((best_vit_id_loaded, best_res_id_loaded), np.nan)
            axes3[1].set_title(f"{best_res_id_loaded}\n(Spearman ρ vs ViT = {rdm_corr_val:.3f})", fontsize=12)
            axes3[1].set_xticks([]); axes3[1].set_yticks([])
            fig3.colorbar(im2, ax=axes3[1], fraction=0.046, pad=0.04)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(Config.RESULTS_DIR / "P3_rdm_comparison.png", dpi=300)
            print("✓ Plot 3 saved.")
            plt.show()
        else:
            print("   ⚠️ Skipping RDM comparison plot - RDMs for best ViT and ResNet configurations not both available.")
    except Exception as e:
        print(f"   ⚠️ Failed to generate Plot 3: {e}")


    # --- Plot 4: Calibration Plots (Reliability Diagrams) ---
    print(f"\n   Generating Plot 4: Calibration Plots...")
    try:
        num_cal_plots = len(valid_model_ids)
        if num_cal_plots > 0:
            fig4, axes4 = plt.subplots(1, num_cal_plots, figsize=(6 * num_cal_plots, 6), sharey=True)
            if num_cal_plots == 1: axes4 = [axes4] # Make axes iterable for single plot
            fig4.suptitle("Model Calibration (Reliability Diagrams)", fontsize=16, fontweight='bold')

            for idx, model_id in enumerate(valid_model_ids):
                cal_data = calibration_data[model_id]
                ax = axes4[idx]
                config_name = '_'.join(model_id.split('_')[:2]) # Base config name for color/style

                # Plot bars: Use bin centers (confidence) vs bin accuracy
                bin_centers = cal_data['bin_lowers'] + (0.5 / Config.CALIBRATION_BINS)
                bar_width = 0.9 / Config.CALIBRATION_BINS

                # Bars showing accuracy in each bin
                ax.bar(bin_centers, cal_data['bin_accuracies'], width=bar_width, alpha=0.7,
                       color=Config.COLORS.get(config_name, '#CCCCCC'), edgecolor='black', label='Accuracy')
                # Optional: Bars showing the gap (confidence - accuracy)
                # ax.bar(bin_centers, cal_data['bin_confidences'] - cal_data['bin_accuracies'],
                #        bottom=cal_data['bin_accuracies'], width=bar_width, alpha=0.5, color='red', edgecolor=None, label='Gap')

                ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration') # Diagonal line

                ax.set_xlabel("Confidence", fontsize=12)
                if idx == 0: ax.set_ylabel("Accuracy", fontsize=12)
                ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
                ax.set_title(f"{model_id}\nECE = {cal_data['ece']:.4f}", fontsize=14)
                ax.legend(fontsize=10, loc='upper left')
                ax.grid(True, linestyle='--', alpha=0.6)
                ax.tick_params(axis='both', which='major', labelsize=10)

            plt.tight_layout(rect=[0, 0.03, 1, 0.94]) # Adjust layout
            plt.savefig(Config.RESULTS_DIR / "P4_calibration.png", dpi=300)
            print("✓ Plot 4 saved.")
            plt.show()
        else:
             print("   ⚠️ Skipping Calibration plot - No valid calibration data available.")
    except Exception as e:
        print(f"   ⚠️ Failed to generate Plot 4: {e}")

    # --- Plot 5: Detailed Analysis Dashboards (t-SNE, RDM, Confusion Matrix) ---
    print(f"\n   Generating Plot 5: Detailed Dashboards...")
    try:
        for model_id in valid_model_ids:
            print(f"   Generating dashboard for: {model_id}")
            data = detailed_analysis_data[model_id]
            fig5 = plt.figure(figsize=(22, 6.5)) # Adjusted size slightly
            gs = fig5.add_gridspec(1, 3, width_ratios=[1, 1, 1.2])
            fig5.suptitle(f"Deep Dive Analysis: {model_id}", fontsize=18, fontweight='bold')

            # A) t-SNE Plot
            ax5a = fig5.add_subplot(gs[0, 0])
            print(f"     Calculating t-SNE ({Config.TSNE_SUBSET_SIZE} points)...")
            tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=Config.ANALYSIS_SEED, init='pca', learning_rate='auto')
            features_tsne_subset = data['features'][tsne_subset_indices]
            labels_tsne_subset = data['labels'][tsne_subset_indices]
            tsne_features = tsne.fit_transform(features_tsne_subset)
            # Plot only a subset of classes for clarity
            unique_labels_in_subset = np.unique(labels_tsne_subset)
            classes_to_show_tsne = unique_labels_in_subset[:min(10, len(unique_labels_in_subset))] # Show up to 10 classes
            palette = sns.color_palette("tab10", len(classes_to_show_tsne))
            for i, class_idx in enumerate(classes_to_show_tsne):
                mask = (labels_tsne_subset == class_idx)
                ax5a.scatter(tsne_features[mask, 0], tsne_features[mask, 1], color=palette[i],
                             label=f'{class_names[class_idx]} ({class_idx})', alpha=0.6, s=15)
            ax5a.set_title(f"A) Feature Space (t-SNE, {Config.TSNE_SUBSET_SIZE} points)", fontsize=14, fontweight='bold')
            ax5a.set_xticks([]); ax5a.set_yticks([])
            ax5a.legend(fontsize=8, loc='center left', bbox_to_anchor=(1.05, 0.5), title="Classes")


            # B) RDM Plot
            ax5b = fig5.add_subplot(gs[0, 1])
            if model_id in rdms:
                im_rdm = ax5b.imshow(rdms[model_id], cmap='viridis', aspect='auto')
                ax5b.set_title(f"B) RDM ({Config.RDM_SUBSET_SIZE} points)", fontsize=14, fontweight='bold')
            else:
                # If RDM wasn't computed earlier (e.g., only 1 model loaded), compute it now on the t-SNE subset
                print(f"     Computing RDM for dashboard ({Config.TSNE_SUBSET_SIZE} points)...")
                rdm_dashboard = compute_rdm(features_tsne_subset, model_id + " (Dashboard)")
                im_rdm = ax5b.imshow(rdm_dashboard, cmap='viridis', aspect='auto')
                ax5b.set_title(f"B) RDM ({Config.TSNE_SUBSET_SIZE} points)", fontsize=14, fontweight='bold')
            ax5b.set_xticks([]); ax5b.set_yticks([])
            fig5.colorbar(im_rdm, ax=ax5b, fraction=0.046, pad=0.04)


            # C) Confusion Matrix Plot
            ax5c = fig5.add_subplot(gs[0, 2])
            cm = confusion_matrix(data['labels'], data['predictions'])
            cm_subset = cm[:Config.CM_SUBSET_CLASSES, :Config.CM_SUBSET_CLASSES]
            class_names_subset = class_names[:Config.CM_SUBSET_CLASSES]
            sns.heatmap(cm_subset, ax=ax5c, cmap='Blues', annot=True, fmt='d', cbar=True,
                        xticklabels=class_names_subset, yticklabels=class_names_subset, annot_kws={"size": 8})
            ax5c.set_title(f"C) Confusion Matrix (Top {Config.CM_SUBSET_CLASSES} Classes)", fontsize=14, fontweight='bold')
            ax5c.tick_params(axis='x', rotation=90, labelsize=8)
            ax5c.tick_params(axis='y', rotation=0, labelsize=8)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(Config.RESULTS_DIR / f"P5_dashboard_{model_id}.png", dpi=300)
            print(f"✓ Dashboard saved for {model_id}.")
            plt.show()

    except Exception as e:
        print(f"   ⚠️ Failed to generate Plot 5 (Dashboards): {e}")


    # --- F. Print Detailed Classification Reports ---
    print("\n" + "="*80 + "\n📋 DETAILED CLASSIFICATION REPORTS (Representative Models)\n" + "="*80)
    try:
        if not valid_model_ids:
            print("   No valid model data available to generate reports.")
        else:
            for model_id in valid_model_ids:
                print(f"\n--- Report for: {model_id} ---")
                data = detailed_analysis_data[model_id]
                # Calculate accuracy from predictions if needed, otherwise use CSV? Best to recalc here.
                acc = accuracy_score(data['labels'], data['predictions']) * 100
                print(f"   Accuracy: {acc:.2f}%")
                # Use target_names for readable labels
                print(classification_report(data['labels'], data['predictions'], target_names=class_names, zero_division=0))
                print("-" * 80)
    except Exception as e:
        print(f"   ⚠️ Failed to generate Classification Reports: {e}")

    print("\n✅ Analysis Complete.")
    print("=" * 80)