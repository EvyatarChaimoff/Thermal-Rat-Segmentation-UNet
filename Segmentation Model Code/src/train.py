import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import os
import numpy as np
from analyze_data import save_fold_distribution
import matplotlib.pyplot as plt
import config 
import utils
from dataset import RatThermalDataset, get_train_transforms, get_val_transforms
# --- CHANGE: Import the class directly ---
from model import RatThermalSegmentor 
# -----------------------------------------
from test import evaluate_fold_test_set 
import json

# --- Directory Setup ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
MODELS_DIR = os.path.join(BASE_DIR, '..', 'models')
METRICS_DIR = os.path.join(BASE_DIR, '..', 'metrics')
KFOLDS_MODEL_PATH = os.path.join(MODELS_DIR, 'kfolds_train')
BASELINE_MODEL_PATH = os.path.join(MODELS_DIR, 'baseline_train')

os.makedirs(KFOLDS_MODEL_PATH, exist_ok=True)
os.makedirs(BASELINE_MODEL_PATH, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)
# -------------------------

def run_training(df):
    """
    Runs 5-Fold Nested Cross-Validation.
    Each Fold has: Train (~60%), Validation (~20%), Test (~20%).
    """
    print(f"Total samples for Cross-Validation: {len(df)}")
    
    if config.K_FOLDS < 2:
        print("[ERROR] K_FOLDS must be >= 2")
        return

    # 1. Outer Split: Separates TEST set from the rest (Train+Val)
    gkf_outer = GroupKFold(n_splits=config.K_FOLDS)
    
    X = df 
    y = df['rat_unique_id'] 
    groups = df['rat_unique_id']

    all_folds_metrics = []

    # --- OUTER LOOP: Defines TEST Set ---
    for fold, (train_val_idx, test_idx) in enumerate(gkf_outer.split(X, y, groups)):
        print(f"\n" + "#"*40)
        print(f"--- FOLD {fold+1}/{config.K_FOLDS} ---")
        print("#"*40)
        
        df_test = df.iloc[test_idx].reset_index(drop=True)
        df_rest = df.iloc[train_val_idx].reset_index(drop=True) # Contains Train + Val candidates
        
        test_rats = df_test['rat_unique_id'].unique().tolist()
        rats_filename = os.path.join(KFOLDS_MODEL_PATH, f"fold_{fold+1}_test_rats.json")
        with open(rats_filename, 'w') as f:
            json.dump(test_rats, f)
        print(f"[INFO] Saved test rats for Fold {fold+1} to {rats_filename}")

        # 2. Inner Split: Separates VALIDATION set from TRAIN set
        gss_inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        rest_groups = df_rest['rat_unique_id']
        
        train_idx, val_idx = next(gss_inner.split(df_rest, groups=rest_groups))
        
        df_train = df_rest.iloc[train_idx].reset_index(drop=True)
        df_val = df_rest.iloc[val_idx].reset_index(drop=True)
        
        # 3. Save Distribution Histogram (Train / Val / Test)
        save_fold_distribution(df_train, df_val, df_test, fold_num=fold+1)
        
        print(f"Train Rats: {sorted(df_train['rat_unique_id'].unique())}")
        print(f"Val Rats:   {sorted(df_val['rat_unique_id'].unique())}")
        print(f"Test Rats:  {sorted(df_test['rat_unique_id'].unique())}")

        # --- Training Setup ---
        train_dataset = RatThermalDataset(df_train, transforms=get_train_transforms())
        val_dataset = RatThermalDataset(df_val, transforms=get_val_transforms())
        
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
        
        # --- CHANGE: Instantiate Class Directly ---
        model = RatThermalSegmentor().to(config.DEVICE)
        # ------------------------------------------
        
        class_weights_calculated = torch.tensor(
            [config.BACKGROUND_WEIGHT,config.HEAD_WEIGHT,config.BODY_WEIGHT, config.TAIL_WEIGHT], 
            dtype=torch.float32
        ).to(config.DEVICE)

        loss_fn = utils.WeightedCombinedDiceFocalLoss(class_weights=class_weights_calculated).to(config.DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE_STAGE1)
        
        # Early Stopping Vars
        patience = config.PATIENCE
        min_val_loss = float('inf')
        epochs_no_improve = 0
        best_miou = -1 
        current_best_path = os.path.join(KFOLDS_MODEL_PATH, f"model_fold_{fold+1}_best.pth")
        
        train_loss_history, val_loss_history, val_miou_history, val_per_class_iou_history = [], [], [], []
        all_batch_train_losses = []

        # --- STAGE 1 ---
        print("--- STAGE 1 ---")
        # Access encoder via 'base_model' because of the new wrapper
        for param in model.base_model.encoder.parameters(): param.requires_grad = False

        for epoch in range(config.EPOCHS_STAGE1):
            epoch_loss, batch_losses = train_one_epoch(model, train_loader, optimizer, loss_fn, config.DEVICE)
            val_loss, val_miou, val_class_iou = validate_one_epoch(model, val_loader, loss_fn, config.DEVICE)
            
            all_batch_train_losses.extend(batch_losses)
            train_loss_history.append(epoch_loss)
            val_loss_history.append(val_loss)
            val_miou_history.append(val_miou)
            val_per_class_iou_history.append(val_class_iou)
            
            print(f"\nEpoch {epoch+1} (S1): Loss: {epoch_loss:.4f}, Val mIoU: {val_miou:.4f}")

            if val_miou > best_miou:
                best_miou = val_miou
                torch.save(model.state_dict(), current_best_path)

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print("Early Stopping (S1)")
                    break

        # --- STAGE 2 ---
        print("--- STAGE 2 ---")
        
        if os.path.exists(current_best_path):
            model.load_state_dict(torch.load(current_best_path))
            print("[INFO] Loaded best model from Stage 1 for Fine-Tuning.")
            
        # Reset Early Stopping variables for Stage 2
        epochs_no_improve = 0
        min_val_loss = float('inf') 
        
        # Unfreeze all parameters
        for param in model.parameters(): param.requires_grad = True
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE_STAGE2)

        for epoch in range(config.EPOCHS_STAGE2):
            epoch_loss, batch_losses = train_one_epoch(model, train_loader, optimizer, loss_fn, config.DEVICE)
            val_loss, val_miou, val_class_iou = validate_one_epoch(model, val_loader, loss_fn, config.DEVICE)
            
            all_batch_train_losses.extend(batch_losses)
            train_loss_history.append(epoch_loss)
            val_loss_history.append(val_loss)
            val_miou_history.append(val_miou)
            val_per_class_iou_history.append(val_class_iou)
            
            total_epoch = epoch + config.EPOCHS_STAGE1
            print(f"Epoch {total_epoch+1} (S2): Loss: {epoch_loss:.4f}, Val mIoU: {val_miou:.4f}")

            if val_miou > best_miou:
                best_miou = val_miou
                torch.save(model.state_dict(), current_best_path)

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print("Early Stopping (S2)")
                    break
        
        # --- Generate Detailed Plot for this Fold ---
        utils.plot_training_progress(
            all_batch_train_losses, val_loss_history, val_miou_history, 
            val_per_class_iou_history, model_type='kfolds', fold=fold+1
        )

        # --- Evaluate on TEST set of this fold ---
        print(f"\n[EVAL] Loading best model for Fold {fold+1} Test Evaluation...")
        
        # Load the best model wrapper
        model = RatThermalSegmentor().to(config.DEVICE)
        model.load_state_dict(torch.load(current_best_path))
        
        test_miou, test_per_class = evaluate_fold_test_set(model, df_test, fold_num_or_name=fold+1)
        
        print(f"Fold {fold+1} Final TEST mIoU: {test_miou:.4f}")

        # Save Metrics CSV
        per_class_arr = np.array(val_per_class_iou_history)
        fold_df = pd.DataFrame({
            'fold': fold + 1,
            'epoch': range(1, len(train_loss_history)+1),
            'train_loss': train_loss_history,
            'val_loss': val_loss_history,
            'val_miou': val_miou_history,
            'test_miou_final': test_miou, 
            'val_iou_bg': per_class_arr[:, 0] if len(per_class_arr)>0 else 0,
            'val_iou_head': per_class_arr[:, 1] if len(per_class_arr)>0 else 0,
            'val_iou_body': per_class_arr[:, 2] if len(per_class_arr)>0 else 0,
            'val_iou_tail': per_class_arr[:, 3] if len(per_class_arr)>0 else 0
        })
        all_folds_metrics.append(fold_df)

    if all_folds_metrics:
        final_df = pd.concat(all_folds_metrics)
        final_df.to_csv(os.path.join(METRICS_DIR, "metrics_kfolds.csv"), index=False)
        print("Saved K-Fold metrics.")


def baseline_training(df):
    """
    Single run: 3-way split (Train 60% / Val 20% / Test 20%)
    """
    print(f"Total samples: {len(df)}")
    
    # 1. Split Test (20%) vs Rest (80%)
    gss_outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    groups = df['rat_unique_id']
    
    rest_idx, test_idx = next(gss_outer.split(df, groups=groups))
    df_rest = df.iloc[rest_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)
    
    test_rats = df_test['rat_unique_id'].unique().tolist()
    rats_filename = os.path.join(BASELINE_MODEL_PATH, "baseline_test_rats.json")
    with open(rats_filename, 'w') as f:
        json.dump(test_rats, f)

    # 2. Split Rest into Train (75% of rest) vs Val (25% of rest) -> 60/20 split of total
    gss_inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    rest_groups = df_rest['rat_unique_id']
    
    train_idx, val_idx = next(gss_inner.split(df_rest, groups=rest_groups))
    df_train = df_rest.iloc[train_idx].reset_index(drop=True)
    df_val = df_rest.iloc[val_idx].reset_index(drop=True)
    
    save_fold_distribution(df_train, df_val, df_test, fold_num='baseline')
    
    print(f"Train Rats: {sorted(df_train['rat_unique_id'].unique())}")
    print(f"Val Rats:   {sorted(df_val['rat_unique_id'].unique())}")
    print(f"Test Rats:  {sorted(df_test['rat_unique_id'].unique())}")
    
    # --- Setup ---
    train_dataset = RatThermalDataset(df_train, transforms=get_train_transforms())
    val_dataset = RatThermalDataset(df_val, transforms=get_val_transforms())
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
    
    # --- CHANGE: Instantiate Class Directly ---
    model = RatThermalSegmentor().to(config.DEVICE)
    # ------------------------------------------

    class_weights = torch.tensor([config.BACKGROUND_WEIGHT,config.HEAD_WEIGHT,config.BODY_WEIGHT, config.TAIL_WEIGHT], dtype=torch.float32).to(config.DEVICE)
    loss_fn = utils.WeightedCombinedDiceFocalLoss(class_weights=class_weights).to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE_STAGE1)
    
    current_best_path = os.path.join(BASELINE_MODEL_PATH, "model_baseline_best.pth")
    best_miou = -1
    patience = config.PATIENCE
    epochs_no_improve = 0
    min_val_loss = float('inf')
    
    train_loss_history, val_loss_history, val_miou_history, val_per_class_iou_history = [], [], [], []
    all_batch_train_losses = []

    # Stage 1
    print("--- STAGE 1 ---")
    # Access encoder via 'base_model'
    for param in model.base_model.encoder.parameters(): param.requires_grad = False
    
    for epoch in range(config.EPOCHS_STAGE1):
        epoch_loss, batch_losses = train_one_epoch(model, train_loader, optimizer, loss_fn, config.DEVICE)
        val_loss, val_miou, val_class_iou = validate_one_epoch(model, val_loader, loss_fn, config.DEVICE)
        
        all_batch_train_losses.extend(batch_losses)
        train_loss_history.append(epoch_loss)
        val_loss_history.append(val_loss)
        val_miou_history.append(val_miou)
        val_per_class_iou_history.append(val_class_iou)
        
        print(f"\nS1 Ep{epoch+1}: Loss {epoch_loss:.4f} Val mIoU {val_miou:.4f}")
        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(model.state_dict(), current_best_path)
        
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve+=1
            if epochs_no_improve >= patience: break
            
    # --- STAGE 2 ---
    print("--- STAGE 2 ---")
        
    if os.path.exists(current_best_path):
        model.load_state_dict(torch.load(current_best_path))
        print("[INFO] Loaded best model from Stage 1 for Fine-Tuning.")
        
    # Reset Early Stopping variables
    epochs_no_improve = 0
    min_val_loss = float('inf')
    
    # Unfreeze all parameters
    for param in model.parameters(): param.requires_grad = True
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE_STAGE2)

    for epoch in range(config.EPOCHS_STAGE2):
        epoch_loss, batch_losses = train_one_epoch(model, train_loader, optimizer, loss_fn, config.DEVICE)
        val_loss, val_miou, val_class_iou = validate_one_epoch(model, val_loader, loss_fn, config.DEVICE)
        
        all_batch_train_losses.extend(batch_losses)
        train_loss_history.append(epoch_loss)
        val_loss_history.append(val_loss)
        val_miou_history.append(val_miou)
        val_per_class_iou_history.append(val_class_iou)
        
        print(f"S2 Ep{epoch}: Loss {epoch_loss:.4f} Val mIoU {val_miou:.4f}")
        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(model.state_dict(), current_best_path)
        
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve+=1
            if epochs_no_improve >= patience: break
    
    # Plot
    utils.plot_training_progress(
        all_batch_train_losses, val_loss_history, val_miou_history, 
        val_per_class_iou_history, model_type='baseline'
    )
    
    # --- TEST EVAL ---
    print(f"\n[EVAL] Loading best model for Baseline Test Evaluation...")
    # Load wrapper
    model = RatThermalSegmentor().to(config.DEVICE)
    model.load_state_dict(torch.load(current_best_path))
    test_miou, _ = evaluate_fold_test_set(model, df_test, fold_num_or_name='baseline')
    print(f"Baseline Final TEST mIoU: {test_miou:.4f}")

    print("Baseline Training Complete.")

# --- Helper functions ---
def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    loop = tqdm(loader, desc="Training")
    running_loss = 0.0
    batch_losses = [] 
    for images, masks in loop:
        images = images.to(device, dtype=torch.float32)
        masks = masks.to(device, dtype=torch.long)
        optimizer.zero_grad()
        
        # Calling model(images) calls the forward() method
        # which returns the raw logits (NO cleaning), which is correct for loss calculation.
        outputs = model(images) 
        
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()
        loss_val = loss.item()
        batch_losses.append(loss_val) 
        running_loss += loss_val * images.size(0)
        loop.set_postfix(loss=loss_val)
    return running_loss / len(loader.dataset), batch_losses 

def validate_one_epoch(model, loader, loss_fn, device):
    """
    Validation loop:
    We usually validate on RAW output to track model convergence cleanly.
    Post-processing (blob cleaning) is applied at inference/test time.
    """
    model.eval()
    loop = tqdm(loader, desc="Validation")
    running_loss = 0.0
    running_miou = 0.0
    running_per_class_iou = torch.zeros(config.NUM_CLASSES).to(device)
    with torch.no_grad():
        for images, masks in loop:
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.long)
            
            # Use raw forward pass for validation loss and metrics
            outputs = model(images) 
            
            loss = loss_fn(outputs, masks)
            miou, per_class_iou = utils.calculate_metrics(outputs, masks, num_classes=config.NUM_CLASSES)
            
            running_loss += loss.item() * images.size(0)
            running_miou += miou.item() * images.size(0)
            running_per_class_iou += per_class_iou.to(device) * images.size(0)
            loop.set_postfix(val_loss=loss.item(), val_miou=miou.item())
    dataset_len = len(loader.dataset)
    return running_loss / dataset_len, running_miou / dataset_len, (running_per_class_iou / dataset_len).cpu().tolist()