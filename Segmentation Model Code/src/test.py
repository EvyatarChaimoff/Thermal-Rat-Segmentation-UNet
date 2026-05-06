import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import cv2 
import numpy as np
from sklearn.metrics import confusion_matrix
import json # For saving detailed metrics
import matplotlib.pyplot as plt
import random
# Import modules from the project structure
import config 
import utils
# --- CHANGE: Import the class directly ---
from model import RatThermalSegmentor
# -----------------------------------------
from dataset import RatThermalDataset, get_val_transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import seaborn as sns
from typing import Optional

# --- Directory Setup (must match train.py) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, '..', 'models')
KFOLDS_MODEL_PATH = os.path.join(MODELS_DIR, 'kfolds_train')
BASELINE_MODEL_PATH = os.path.join(MODELS_DIR, 'baseline_train')
TEST_METRICS_PATH = os.path.join(BASE_DIR, '..', 'metrics', 'test')
TEST_FIGURE_DIR_BASELINE = os.path.join(BASE_DIR,'..','test preds figures','baseline')
TEST_FIGURE_DIR_KFOLDS = os.path.join(BASE_DIR, '..', 'test preds figures', 'kfolds')
# Ensure metrics directory exists
os.makedirs(TEST_METRICS_PATH, exist_ok=True)
# ---------------------------------------------

def load_test_data(df, model_type='baseline', fold=None):
    """Filters metadata based on saved JSON rat lists from training."""
    rats_file = None
    if model_type == 'baseline':
        rats_file = os.path.join(BASELINE_MODEL_PATH, "baseline_test_rats.json")
    elif model_type == 'kfolds' and fold is not None:
        # Match naming from train.py: fold_{fold}_test_rats.json
        rats_file = os.path.join(KFOLDS_MODEL_PATH, f"fold_{fold}_test_rats.json")

    if rats_file and os.path.exists(rats_file):
        with open(rats_file, 'r') as f:
            test_rats = json.load(f)
        print(f"[INFO] Loading saved test rats for {model_type}: {test_rats}")
    else:
        print(f"[ERROR] No saved rat list found at {rats_file}. Check training outputs.")
        return None, None, None

    df_test = df[df['rat_unique_id'].isin(test_rats)].reset_index(drop=True)
    test_dataset = RatThermalDataset(df_test, transforms=get_val_transforms())
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
    
    return test_loader, df_test, test_dataset

def load_best_model(model_type, fold=None):
    """
    Loads the best trained model based on the type (baseline or kfolds).
    :param model_type: 'baseline' or 'kfolds'
    :param fold: Required only for 'kfolds' (e.g., the best model from a specific fold, or None for baseline)
    """
    # --- CHANGE: Instantiate Class Directly ---
    model = RatThermalSegmentor().to(config.DEVICE)
    # ------------------------------------------
    device = config.DEVICE
    
    if model_type == 'baseline':
        model_path = os.path.join(BASELINE_MODEL_PATH, "model_baseline_best.pth")
    elif model_type == 'kfolds':
        if fold is None:
             print("[ERROR] For K-Folds testing, you must specify the fold number (or implement averaging).")
             return None
        model_path = os.path.join(KFOLDS_MODEL_PATH, f"model_fold_{fold}_best.pth")
    else:
        print("[ERROR] Invalid model_type. Must be 'baseline' or 'kfolds'.")
        return None

    if not os.path.exists(model_path):
        print(f"[CRITICAL ERROR] Model file not found at: {model_path}")
        return None
        
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"SUCCESS: Loaded model from {model_path}")
        return model
    except Exception as e:
        print(f"[CRITICAL ERROR] Failed to load model state dict: {e}")
        print("NOTE: If you changed the model architecture (added wrapper), you must RETRAIN the model.")
        return None


def run_test_evaluation(df, model_type='baseline', fold=1):
    """
    Runs the final evaluation on the dedicated test set.
    Calculates metrics (mIoU, CM, Per-Class IoU) using the CLEANED predictions.
    
    :param df: Full metadata DataFrame.
    :param model_type: 'baseline' or 'kfolds'.
    :param fold: If model_type='kfolds', which fold's best model to use.
    """
    print(f"\n--- Starting Test Evaluation using {model_type.upper()} model ---")

    test_loader, df_test, test_dataset = load_test_data(df, model_type=model_type, fold=fold)
    if test_loader is None:
        return

    model = load_best_model(model_type, fold=fold)
    if model is None:
        return

    device = config.DEVICE
    
    total_samples = len(df_test)
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Testing"):
            images = images.to(device, dtype=torch.float32)
            # masks are already tensor, move to cpu later
            
            # --- USE PREDICT_AND_CLEAN METHOD ---
            # This returns a numpy array of shape (B, H, W) with cleaned class indices
            cleaned_preds = model.predict_and_clean(images) 
            
            # Prepare data for detailed per-class metrics
            # Flatten predictions and targets
            preds_flat = cleaned_preds.flatten()
            targets_flat = masks.numpy().flatten() # Assuming masks is (B, H, W)
            
            all_preds.extend(preds_flat.tolist())
            all_targets.extend(targets_flat.tolist())

    # Calculate global metrics on all pixels
    all_preds_np = np.array(all_preds)
    all_targets_np = np.array(all_targets)
    
    # Calculate per-class IoU and mIoU
    per_class_iou = calculate_per_class_iou(all_targets_np, all_preds_np, config.NUM_CLASSES)
    final_miou = np.mean(per_class_iou)
    
    cm = confusion_matrix(all_targets_np, all_preds_np, labels=range(config.NUM_CLASSES))
    
    # --- Report & Saving ---
    report = {
        'model_type': model_type,
        'fold': fold if model_type == 'kfolds' else 'N/A',
        'mean_iou': final_miou,
        'per_class_iou': {f'class_{i}': iou for i, iou in enumerate(per_class_iou)},
        'confusion_matrix': cm.tolist(),
        'num_samples': total_samples
    }
    
    # Print summary
    print("\n" + "="*50)
    print(f"FINAL TEST REPORT ({model_type.upper()})")
    print("="*50)
    print(f"Overall Mean IoU (mIoU): {final_miou:.4f}")
    print("Per Class IoU:")
    class_names = ['Background', 'Head', 'Body', 'Tail']
    for i, iou in enumerate(per_class_iou):
        print(f"  {class_names[i]:<10}: {iou:.4f}")
    print("="*50)

    # --- 1. JSON Saving Logic (Metrics) ---
    if model_type == 'baseline':
        save_dir = os.path.join(TEST_METRICS_PATH, 'baseline')
        json_file_name = "test_metrics_baseline.json"
        plot_file_name = "confusion_matrix_baseline.png"
    elif model_type == 'kfolds':
        kfolds_dir = os.path.join(TEST_METRICS_PATH, 'kfolds')
        save_dir = os.path.join(kfolds_dir, f'fold_{fold}')
        json_file_name = f"test_metrics_kfolds_fold_{fold}.json"
        plot_file_name = f"confusion_matrix_kfolds_fold_{fold}.png"
    else:
        # Default fallback
        save_dir = TEST_METRICS_PATH
        json_file_name = f"test_metrics_{model_type}_default.json"
        plot_file_name = f"confusion_matrix_{model_type}_default.png"

    # Ensure the directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Save JSON
    json_metrics_path = os.path.join(save_dir, json_file_name)
    with open(json_metrics_path, 'w') as f:
        json.dump(report, f, indent=4)
        
    print(f"SUCCESS: Test metrics saved to '{json_metrics_path}'")
    
    # --- 2. Plot Saving Logic (Confusion Matrix) ---
    plot_cm_path = os.path.join(save_dir, plot_file_name)
    plot_and_save_confusion_matrix(cm, class_names, plot_cm_path)


def calculate_per_class_iou(targets_flat, preds_flat, num_classes):
    """Calculates IoU for each class separately (required for the report)."""
    
    # Calculate confusion matrix
    cm = confusion_matrix(targets_flat, preds_flat, labels=range(num_classes))
    
    # Extract TP, FP, FN for each class
    # IoU = TP / (TP + FP + FN)
    
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

def denormalize_image(tensor_img):
    """Converts a normalized PyTorch tensor back to a displayable numpy image (H, W, C or H, W)."""
    # Assuming input is (C, H, W) tensor
    img = tensor_img.permute(1, 2, 0).cpu().numpy()
    # Take the first channel for the 2D thermal image display
    thermal_2d = img[:, :, 0]
    # Clip back to 0-1 range (should already be there due to initial normalization)
    return np.clip(thermal_2d, 0, 1)

def visualize_test_sample(
    df_test_or_dataset,
    model,
    device,
    model_type: str = 'baseline',
    numFolds: int = 1,
    saveFigure: bool = True,
    fileName: Optional[str] = None,
    random_sample: bool = True,
    idx: int = 0
):
    """
    Visualizes a single test sample with model prediction (CLEANED).
    """

    # Determine input type
    is_dataframe = isinstance(df_test_or_dataset, type(None)) == False and isinstance(df_test_or_dataset, pd.DataFrame)

    # Handle DataFrame input
    if is_dataframe:
        df_test = df_test_or_dataset
        n_samples = len(df_test)
        if n_samples == 0:
            print("[WARNING] Test DataFrame is empty. Cannot visualize.")
            return
    else:
        # Assume a RatThermalDataset
        test_dataset = df_test_or_dataset
        try:
            n_samples = len(test_dataset)
        except Exception:
            print("[ERROR] Provided dataset object has no __len__(). Cannot visualize.")
            return
        if n_samples == 0:
            print("[WARNING] Test Dataset is empty. Cannot visualize.")
            return

    # Choose sample index
    if random_sample:
        chosen_idx = random.randint(0, n_samples - 1)
    else:
        if idx < 0 or idx >= n_samples:
            print(f"[WARNING] Index {idx} is out of bounds for test set of size {n_samples}.")
            return
        chosen_idx = idx

    # Build single-sample dataset and get sample info
    if is_dataframe:
        sample_df = df_test.iloc[[chosen_idx]].reset_index(drop=True)
        single_dataset = RatThermalDataset(sample_df, transforms=get_val_transforms())
        rat_id = sample_df['rat_unique_id'].iloc[0] if 'rat_unique_id' in sample_df.columns else 'N/A'
        image, mask_gt = single_dataset[0]  # single sample
    else:
        single_dataset = test_dataset
        rat_id = getattr(test_dataset, 'df', None)
        if hasattr(test_dataset, 'df') and isinstance(test_dataset.df, pd.DataFrame):
            rat_id = test_dataset.df['rat_unique_id'].iloc[chosen_idx]
        else:
            rat_id = 'N/A'
        image, mask_gt = single_dataset[chosen_idx]

    # Move model to device
    model.to(device)
    model.eval()
    
    # --- USE PREDICT_AND_CLEAN METHOD ---
    # predict_and_clean expects a batch, so we unsqueeze the single image
    image_input = image.unsqueeze(0).to(device, dtype=torch.float32)  # (1,C,H,W)
    
    # The method returns numpy array (B, H, W)
    cleaned_preds_batch = model.predict_and_clean(image_input)
    mask_pred = cleaned_preds_batch[0] # Take the first (and only) item
    # ----------------------------------------

    # Convert tensors to numpy arrays for plotting
    thermal_2d = image.cpu().numpy()  # shape (C,H,W)

    # Handle channel dimension for plotting
    if thermal_2d.ndim == 3:
        if thermal_2d.shape[0] == 1:  # single-channel, squeeze
            thermal_2d = thermal_2d.squeeze(0)
        elif thermal_2d.shape[0] == 3:  # RGB, convert to HWC
            thermal_2d = np.transpose(thermal_2d, (1, 2, 0))

    mask_gt_np = mask_gt.cpu().numpy() if isinstance(mask_gt, torch.Tensor) else np.array(mask_gt)
    if mask_gt_np.ndim == 3 and mask_gt_np.shape[0] == 1:
        mask_gt_np = mask_gt_np.squeeze(0)

    # Setup colormap and class names
    num_classes = config.NUM_CLASSES
    cmap = plt.cm.get_cmap('jet', num_classes)
    class_names = getattr(config, 'CLASS_NAMES', [f"class_{i}" for i in range(num_classes)])

    # Plot input, GT, prediction, overlays
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"Test Sample Visualization (Rat ID: {rat_id}) - WITH BLOB CLEANING", fontsize=16)

    axes[0, 0].imshow(thermal_2d, cmap='inferno')
    axes[0, 0].set_title("Thermal Image (Input)")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(mask_gt_np, cmap=cmap, vmin=0, vmax=num_classes - 1)
    axes[0, 1].set_title("Ground Truth Mask")
    axes[0, 1].axis('off')

    axes[0, 2].imshow(mask_pred, cmap=cmap, vmin=0, vmax=num_classes - 1)
    axes[0, 2].set_title("Model Prediction (Cleaned)")
    axes[0, 2].axis('off')

    axes[1, 0].imshow(thermal_2d, cmap='inferno')
    axes[1, 0].imshow(mask_gt_np, cmap=cmap, alpha=0.5)
    axes[1, 0].set_title("Overlay: Thermal + GT")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(thermal_2d, cmap='inferno')
    axes[1, 1].imshow(mask_pred, cmap=cmap, alpha=0.5)
    axes[1, 1].set_title("Overlay: Thermal + Prediction (Cleaned)")
    axes[1, 1].axis('off')

    axes[1, 2].set_visible(False)

    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(axes[0, 1].images[0], cax=cbar_ax, ticks=list(range(num_classes)))
    cbar.ax.set_yticklabels(class_names)

    plt.tight_layout(rect=[0, 0, 0.9, 1])

    # Save or show
    if saveFigure:
        if not fileName:
            print("[WARNING] fileName must be provided when saveFigure=True. Figure not saved.")
            plt.close(fig)
            return

        save_dir = TEST_FIGURE_DIR_BASELINE if model_type == 'baseline' else os.path.join(TEST_FIGURE_DIR_KFOLDS, f'fold_{numFolds}')
        os.makedirs(save_dir, exist_ok=True)
        figure_path = os.path.join(save_dir, f"{fileName}.png")
        try:
            plt.savefig(figure_path, bbox_inches='tight')
            print(f"[SUCCESS] Figure saved to '{figure_path}'")
        except Exception as e:
            print(f"[ERROR] Failed to save figure: {e}")
        finally:
            plt.close(fig)
    else:
        plt.show()
        plt.close(fig)

def run_test_visualization(df, model_type='baseline', fold=1):
    """
    Loads the trained model and displays a random segmentation visualization 
    from the test set (Option 6).
    """
    print(f"\n--- Starting Test Visualization using {model_type.upper()} model ---")

    # Load data needed for visualization (df_test)
    _, df_test,_ = load_test_data(df)
    if df_test.empty:
        print("[ERROR] Test data not found or empty.")
        return

    # Load the best model
    model = load_best_model(model_type, fold=fold)
    if model is None:
        return
        
    device = config.DEVICE
    
    # Call the visualization function
    print("--- Displaying random segmented sample from Test Set ---")
    visualize_test_sample(df_test, model, device,saveFigure=False)

def preprocess_single_csv(csv_path):
    """
    Loads, rotates, and normalizes a single CSV temperature matrix, 
    matching the logic in RatThermalDataset.
    Returns: torch tensor image (C, H, W) and numpy array image (H, W, 3) 
    """
    # 1. Load CSV (Temperature Matrix) - Rows 3+, Columns 2+
    try:
        temp_df = pd.read_csv(csv_path, header=None, skiprows=2)
        # Assuming column 0 is index/time, we take all subsequent columns
        temp_matrix = temp_df.iloc[:, 1:].values.astype(np.float32)
    except Exception as e:
        print(f"Error loading CSV at {csv_path}: {e}")
        temp_matrix = np.zeros((config.IMG_HEIGHT, config.IMG_WIDTH), dtype=np.float32)
        return torch.zeros(3, config.IMG_HEIGHT, config.IMG_WIDTH), np.zeros((config.IMG_HEIGHT, config.IMG_WIDTH, 3))

    # 2. Rotation Logic (Apply rotation if width > height to standardize to 640x480)
    h, w = temp_matrix.shape
    if h < w: 
        print(f"[INFO] Rotating {h}x{w} matrix to standardize format.")
        temp_matrix = cv2.rotate(temp_matrix, cv2.ROTATE_90_CLOCKWISE)
    
    # After rotation, ensure we have the target shape (640, 480) if loaded correctly
    
    # 3. Normalization (Per-image normalization)
    min_val = np.min(temp_matrix)
    max_val = np.max(temp_matrix)
    
    if max_val > min_val:
        image = (temp_matrix - min_val) / (max_val - min_val)
    else:
        image = np.zeros_like(temp_matrix)

    # 4. Replicate to 3 channels for ResNet encoder input (H, W, 3)
    image_h_w_c = np.stack([image, image, image], axis=-1) 
    
    # 5. Apply Resize and ToTensorV2 (validation transforms)
    transforms = A.Compose([
        A.Resize(height=config.IMG_HEIGHT, width=config.IMG_WIDTH, interpolation=cv2.INTER_NEAREST),
        ToTensorV2() # Returns C, H, W
    ])
    
    transformed = transforms(image=image_h_w_c)
    
    # Return torch tensor (C, H, W) and the original normalized numpy array (H, W, 3)
    return transformed['image'], image_h_w_c

def run_single_inference(model_type, fold=1):
    """
    Opens a file dialog, loads a single CSV, runs the model inference, 
    and displays the segmentation result.
    """
    try:
        # Import the GUI component ONLY when needed
        from tkinter import Tk
        from tkinter.filedialog import askopenfilename
    except ImportError:
        print("[ERROR] Tkinter (GUI library) not found. Cannot open file dialog.")
        return

    # 1. Initialize Tkinter and open file dialog
    Tk().withdraw() # We don't want the full GUI main window to appear
    csv_path = askopenfilename(
        title="Select a Rat Thermal CSV File",
        filetypes=[("CSV files", "*.csv")]
    )
    
    if not csv_path:
        print("[INFO] File selection cancelled.")
        return

    print(f"Selected file: {csv_path}")
    
    # 2. Load the Model
    model = load_best_model(model_type, fold=fold)
    if model is None:
        return

    # 3. Preprocess and prepare data
    image_tensor, image_numpy = preprocess_single_csv(csv_path)
    
    device = config.DEVICE
    model.eval()
    
    # Add batch dimension and move to device
    image_input = image_tensor.unsqueeze(0).to(device, dtype=torch.float32) 
    
    # 4. Run Inference with CLEANING
    # Use the new method directly
    cleaned_preds_batch = model.predict_and_clean(image_input)
    mask_pred = cleaned_preds_batch[0] # (H, W)
        
    # 5. Visualization (Simplified version of visualize_test_sample)
    
    thermal_2d = image_numpy[:, :, 0] # Extract the 2D thermal channel
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    file_name = os.path.basename(csv_path)
    
    # Define color map properties
    cmap = plt.cm.get_cmap('jet', config.NUM_CLASSES)
    
    # --- Plot 1: Thermal Image ---
    axes[0].imshow(thermal_2d, cmap='inferno') 
    axes[0].set_title(f"Thermal Input ({file_name})")
    axes[0].axis('off')

    # --- Plot 2: Model Prediction Mask ---
    im_pred = axes[1].imshow(mask_pred, cmap=cmap, vmin=0, vmax=config.NUM_CLASSES - 1, interpolation='nearest')
    axes[1].set_title("Model Prediction Mask (Cleaned)")
    axes[1].axis('off')
    
    # --- Plot 3: Overlay ---
    axes[2].imshow(thermal_2d, cmap='inferno')
    axes[2].imshow(mask_pred, cmap=cmap, vmin=0, vmax=config.NUM_CLASSES - 1, alpha=0.5, interpolation='nearest')
    axes[2].set_title("Overlay: Thermal + Prediction")
    axes[2].axis('off')

    # Colorbar Setup (shared)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) 
    cbar = plt.colorbar(im_pred, cax=cbar_ax, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(['Background', 'Head', 'Body', 'Tail'])
    
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()

def plot_and_save_confusion_matrix(cm, class_names, output_path):
    """
    Plots the Confusion Matrix using seaborn and saves the figure.
    """
    plt.figure(figsize=(8, 7))
    sns.heatmap(
        cm, 
        annot=True,              # Display raw numbers
        fmt='d',                 # Format as integer (raw counts)
        cmap='Blues',            # Color scheme
        xticklabels=class_names, 
        yticklabels=class_names,
        linewidths=0.5,
        linecolor='black'
    )
    plt.title('Confusion Matrix (Raw Counts)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    try:
        plt.savefig(output_path)
        print(f"SUCCESS: Confusion Matrix plot saved to '{output_path}'")
    except Exception as e:
        print(f"[ERROR] Failed to save CM plot: {e}")
    finally:
        plt.close() # Close the figure to free memory

def evaluate_fold_test_set(model, df_test, fold_num_or_name):
    """
    Called by train.py to evaluate the model on the fold's specific test set 
    immediately after training.
    """
    model.eval()
    test_dataset = RatThermalDataset(df_test, transforms=get_val_transforms())
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    all_preds = []
    all_targets = []
    
    # This loop is effectively the same as run_test_evaluation but returns vars instead of saving JSON
    for images, masks in test_loader:
        images = images.to(config.DEVICE, dtype=torch.float32)
        
        # --- USE PREDICT_AND_CLEAN METHOD ---
        cleaned_preds = model.predict_and_clean(images) # (B, H, W)
        
        preds_flat = cleaned_preds.flatten()
        targets_flat = masks.numpy().flatten()
        
        all_preds.extend(preds_flat)
        all_targets.extend(targets_flat)
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Calculate Metrics
    # Note: Using confusion matrix based IoU for consistency
    cm = confusion_matrix(all_targets, all_preds, labels=range(config.NUM_CLASSES))
    ious = []
    for i in range(config.NUM_CLASSES):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        iou = tp / (tp + fp + fn + 1e-6)
        ious.append(iou)
    
    miou = np.mean(ious)
    return miou, ious