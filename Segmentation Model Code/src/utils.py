import segmentation_models_pytorch as smp
from segmentation_models_pytorch.metrics.functional import get_stats, iou_score
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn
import pandas as pd
import numpy as np
import config
import os
# --- Loss Function (No Changes) ---
class WeightedCombinedDiceFocalLoss(nn.Module):
    def __init__(self, class_weights=None, dice_weight=0.5, focal_weight=0.5, gamma=2.0, smooth=1e-5):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.class_weights = class_weights
        self.smooth = smooth
        self.gamma = gamma
        
    def forward(self, outputs, targets):
        num_classes = outputs.size(1)
        log_probs = F.log_softmax(outputs, dim=1)
        probs = torch.exp(log_probs)
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        pt = (probs * targets_one_hot).sum(dim=1)
        log_pt = (log_probs * targets_one_hot).sum(dim=1)
        focal_term = (1 - pt).pow(self.gamma)
        loss_per_pixel = - focal_term * log_pt
        if self.class_weights is not None:
            if self.class_weights.device != outputs.device:
                self.class_weights = self.class_weights.to(outputs.device)
            pixel_weights = self.class_weights[targets] 
            loss_per_pixel = loss_per_pixel * pixel_weights
        focal_loss = loss_per_pixel.mean()

        dice_loss = 0.0
        probs_dice = F.softmax(outputs, dim=1)
        total_weight = 0.0
        for c in range(num_classes):
            p_c = probs_dice[:, c, :, :]
            t_c = targets_one_hot[:, c, :, :]
            intersection = (p_c * t_c).sum(dim=(1, 2))
            union = p_c.sum(dim=(1, 2)) + t_c.sum(dim=(1, 2))
            score = (2. * intersection + self.smooth) / (union + self.smooth)
            loss_c = 1.0 - score.mean()
            weight_c = 1.0
            if self.class_weights is not None:
                weight_c = self.class_weights[c].item()
            dice_loss += loss_c * weight_c
            total_weight += weight_c
            
        if self.class_weights is not None:
            dice_loss /= total_weight
        else:
            dice_loss /= num_classes

        return self.dice_weight * dice_loss + self.focal_weight * focal_loss


def calculate_miou(outputs, targets, num_classes=4):
    # Convert logits to predicted labels
    preds = torch.argmax(outputs, dim=1)  # (B,H,W)

    # Get stats: shape (B, C)
    tp, fp, fn, tn = get_stats(
        preds, targets,
        mode='multiclass',
        num_classes=num_classes
    )

    # Now compute IoU
    miou = iou_score(
        tp, fp, fn, tn,
        reduction="macro"   # << זה שקול ל-mIoU
    )

    return miou

# --- Metrics Calculation ---
def calculate_metrics(outputs, targets, num_classes=4):
    """
    Calculates both Mean IoU and Per-Class IoU efficiently using SMP functionals.
    """
    # Convert logits to predicted labels
    preds = torch.argmax(outputs, dim=1)

    # Get stats: tp, fp, fn, tn
    # This works on GPU and is fast
    tp, fp, fn, tn = get_stats(
        preds, targets,
        mode='multiclass',
        num_classes=num_classes
    )

    # 1. Global mIoU (Macro average)
    miou = iou_score(tp, fp, fn, tn, reduction="macro")

    # 2. Per Class IoU
    # reduction="none" returns tensor of shape (Batch_Size, Num_Classes)
    # We compute the mean over the batch dimension to get average IoU per class for this batch
    per_class_iou_batch = iou_score(tp, fp, fn, tn, reduction="none") 
    per_class_iou_mean = torch.mean(per_class_iou_batch, dim=0) # Result: Tensor of size [Num_Classes]

    return miou, per_class_iou_mean

# --- Plotting ---

def plot_training_progress(train_loss_per_batch, val_losses, val_mious, val_per_class_ious, model_type, fold=None):
    """
    Plots Training Loss (per batch, scaled to epochs), Val Loss, Val mIoU, and Per-Class IoU.
    The X-axis represents Epochs for better readability in technical reports.
    """
    os.makedirs(config.TRAINING_GRAPHS_DIR, exist_ok=True)
    
    num_epochs = len(val_losses)
    if num_epochs == 0:
        return

    # Determine file naming based on training mode
    if model_type == 'baseline':
        filename = "training_graph_baseline.png"
        title_suffix = "Baseline"
    else:
        filename = f"training_graph_fold_{fold}.png"
        title_suffix = f"Fold {fold}"
        
    save_path = os.path.join(config.TRAINING_GRAPHS_DIR, filename)

    # Create X-axis scales:
    # Validation metrics occur at integer epoch steps
    epoch_x_axis = np.arange(1, num_epochs + 1)
    # Map batch-level training loss across the continuous epoch range
    batch_x_axis = np.linspace(1, num_epochs, len(train_loss_per_batch))

    # Prepare per-class data for plotting
    per_class_arr = np.array(val_per_class_ious) 
    class_names = ['Background', 'Head', 'Body', 'Tail']
    colors = ['gray', 'red', 'blue', 'orange']

    # Initialize a figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
    # --- Subplot 1: Losses ---
    # Plot raw batch loss with low alpha for background context
    ax1.plot(batch_x_axis, train_loss_per_batch, label='Train Loss (Batch)', color='lightblue', alpha=0.4, linewidth=1)
    
    # Calculate and plot a smoothed trend line for training loss
    window = max(len(train_loss_per_batch) // (num_epochs * 2), 5)
    if len(train_loss_per_batch) > window:
        smoothed_loss = np.convolve(train_loss_per_batch, np.ones(window)/window, mode='valid')
        smoothed_x = np.linspace(1, num_epochs, len(smoothed_loss))
        ax1.plot(smoothed_x, smoothed_loss, label='Train Loss (Trend)', color='blue', linewidth=2)
    
    # Plot validation loss at the end of each epoch
    ax1.plot(epoch_x_axis, val_losses, label='Val Loss (Epoch)', color='red', marker='o', linewidth=2, linestyle='--')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Training & Validation Loss - {title_suffix}')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # --- Subplot 2: mIoU ---
    ax2.plot(epoch_x_axis, val_mious, label='Val mIoU (Mean)', color='green', marker='s', linewidth=2)
    ax2.set_ylabel('mIoU Score')
    ax2.set_title(f'Validation Mean IoU - {title_suffix}')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.0)
    
    # --- Subplot 3: Per-Class IoU ---
    for i in range(config.NUM_CLASSES):
        if per_class_arr.shape[0] > 0:
            ax3.plot(epoch_x_axis, per_class_arr[:, i], label=f'{class_names[i]} IoU', 
                     color=colors[i], marker='.', linewidth=1.5)

    ax3.set_ylabel('IoU Score')
    ax3.set_xlabel('Epochs') 
    ax3.set_title(f'Per-Class Validation IoU - {title_suffix}')
    ax3.legend(loc='upper left', bbox_to_anchor=(1, 1)) 
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1.0)
    
    # Add vertical markers for each epoch to align metrics visually
    for e in epoch_x_axis:
        ax1.axvline(x=e, color='gray', linestyle=':', alpha=0.3)
        ax2.axvline(x=e, color='gray', linestyle=':', alpha=0.3)
        ax3.axvline(x=e, color='gray', linestyle=':', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f"[INFO] Training graph saved (Epoch-based) to: {save_path}")

# --- Other Utils (No Changes) ---
def visualize_data_distribution(df):
    # (Same as your file)
    print("\n--- Visualizing Data Distribution by Rat ID ---")
    if 'rat_unique_id' not in df.columns:
        return
    rat_counts = df['rat_unique_id'].value_counts()
    if rat_counts.empty:
        return
    def extract_rat_number(rat_id):
        try:
            return int(str(rat_id).replace('Rat', ''))
        except:
            return 0 
    sorted_index = rat_counts.index.to_series().apply(extract_rat_number).sort_values().index
    rat_counts = rat_counts.reindex(sorted_index)
    plt.figure(figsize=(12, 6))
    bars = plt.bar(rat_counts.index.astype(str), rat_counts.values, color='skyblue')
    plt.title('Distribution of Images per Rat (RatThermalDataset)', fontsize=14)
    plt.xlabel('Rat Unique ID', fontsize=12)
    plt.ylabel('Number of Samples (Images)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., 1.01*height, '%d' % int(height), ha='center', va='bottom')
    plt.tight_layout()
    plt.show()

def calculate_per_class_iou(targets_flat, preds_flat, num_classes):
    # (Same as your file - kept for test.py)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(targets_flat, preds_flat, labels=range(num_classes))
    ious = []
    for i in range(num_classes):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        denominator = TP + FP + FN
        if denominator == 0:
            iou = 0.0
        else:
            iou = TP / denominator
        ious.append(iou)
    return ious

def load_metadata(csv_path):
    # (Same as your file)
    print("--- Loading Configuration ---")
    print(f"Reading CSV from: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Could not find file at {csv_path}.")
        return None
    print(f"Loaded {len(df)} rows.")
    return df

def generate_all_test_figures(test_dataset, model, model_type, fold_num=1):
    """
    Helper function to iterate over the test dataset and save visualizations.
    Prevents code duplication between manual choice and 'Run All'.
    """
    print(f"\n--- Generating and Saving Test Figures ({model_type.upper()}{f' - Fold {fold_num}' if model_type=='kfolds' else ''}) ---")
    from test import visualize_test_sample
    for i in range(len(test_dataset)):
        try:
            # Extract full path of the mask
            full_path = test_dataset.mask_paths[i]

            # Normalize slashes just in case
            full_path = full_path.replace("\\", "/")

            # Split path parts
            parts = full_path.split("/")

            # Extract "RatX" (Assumes structure like .../Rat1/...)
            rat_folder = parts[-3]       # e.g. "Rat1"
            rat_num = rat_folder.replace("Rat", "")

            # Extract filename: Thermal_X.png -> Thermal_X
            file_name = parts[-1].replace(".png", "")
            if "_" in file_name:
                thermal_num = file_name.split("_")[1]   # X
            else:
                thermal_num = i # Fallback if naming is different

            # Build final clean filename
            file_name_clean = f"Rat {rat_num} Thermal {thermal_num}"

            # Call visualization function
            visualize_test_sample(
                test_dataset, 
                model, 
                config.DEVICE, 
                model_type=model_type, 
                numFolds=fold_num,
                saveFigure=True,
                fileName=file_name_clean, 
                random_sample=False, 
                idx=i
            )
        except Exception as e:
            print(f"[WARNING] Could not save figure for index {i}: {e}")