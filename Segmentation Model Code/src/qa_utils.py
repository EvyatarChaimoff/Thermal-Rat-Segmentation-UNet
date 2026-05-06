import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import os
import random
from dataset import RatThermalDataset


def run_qa_process(df, csv_path):
    """Encapsulates the full QA and interactive cleaning process (Option 1)."""
    if df is None or df.empty:
        print("[ERROR] Cannot run QA. Metadata DataFrame is empty or failed to load.")
        return
        
    # Create dataset for QA (Using get_val_transforms for consistency)
    from dataset import get_val_transforms
    ds = RatThermalDataset(df, transforms=get_val_transforms())
    
    # 1. Run QA and get list of bad samples
    print("\n--- Starting Full Dataset QA Scan ---")
    bad_samples = run_dataset_qa(ds) # Assuming run_dataset_qa is in qa_utils and takes the dataset
    
    # 2. Interactive Cleaning Step
    filter_and_save_clean_csv(bad_samples, csv_path)

    # 3. Visualization (Optional)
    print("\n--- Visualizing 3 Random Samples ---")
    indices = list(range(len(ds)))
    if indices:
        random_indices = random.sample(indices, min(3, len(indices)))
        for i in random_indices:
            visualize_sample(ds, index=i) # Assuming visualize_sample is in qa_utils

def run_dataset_qa(ds):
    """
    Iterates over the entire dataset, checks dimensions, and counts classes.
    Returns: list of tuples (index, path) for samples with missing classes.
    """
    print(f"Starting QA on dataset with {len(ds)} samples...")
    
    count_valid_shapes = 0
    count_full_classes = 0 
    missing_samples_info = [] 
    
    for i in tqdm(range(len(ds))):
        try:
            image, mask = ds[i]
            
            # 1. Check Dimensions
            if image.shape[1] == 640 and image.shape[2] == 480:
                count_valid_shapes += 1
            else:
                print(f"[Error] Index {i} has invalid shape: {image.shape}")

            # 2. Check Labels
            unique_labels = torch.unique(mask).tolist()
            
            if len(unique_labels) == 4:
                count_full_classes += 1
            else:
                # Retrieve path (handling potential attribute naming diffs)
                current_path = ds.mask_paths[i]
                missing_samples_info.append((i, current_path))
                
        except Exception as e:
            print(f"[CRITICAL ERROR] Could not load index {i}: {e}")
            # Consider this a missing sample as well
            missing_samples_info.append((i, "Load Error"))

    # --- Summary Report ---
    print("\n" + "="*40)
    print("DATASET QA REPORT")
    print("="*40)
    print(f"Total Samples Scanned: {len(ds)}")
    print(f"Valid Shapes (640x480): {count_valid_shapes}/{len(ds)}")
    print(f"Samples with ALL 4 classes: {count_full_classes}")
    print(f"Samples with missing classes: {len(missing_samples_info)}")
    print("="*40)

    return missing_samples_info

def filter_and_save_clean_csv(bad_indices_list, original_csv_path):
    """
    Prompts user to create a NEW clean CSV excluding the bad indices.
    Does NOT overwrite the original file.
    """
    if not bad_indices_list:
        print("\n[INFO] Dataset is perfect! No cleaning needed.")
        return

    print(f"\nFound {len(bad_indices_list)} defective samples (less than 4 classes).")
    print("I can create a NEW csv file without these samples.")
    user_input = input(">>> Create new CLEAN metadata file? (The original file will remain untouched) [y/n]: ").strip().lower()

    if user_input in ['y', 'yes']:
        try:
            # 1. Load original CSV
            print(f"Reading original file: {original_csv_path}")
            df = pd.read_csv(original_csv_path)
            initial_len = len(df)
            
            # 2. Drop rows
            indices_to_drop = [item[0] for item in bad_indices_list]
            df_clean = df.drop(index=indices_to_drop).reset_index(drop=True)
            
            # 3. Construct NEW filename based on the old one
            folder = os.path.dirname(original_csv_path)
            filename = os.path.basename(original_csv_path)
            name_without_ext = os.path.splitext(filename)[0]
            
            # New name: metadata_after_QA.csv
            new_filename = "metadata_after_QA.csv"
            output_path = os.path.join(folder, new_filename)
            
            # 4. Save to the NEW path
            df_clean.to_csv(output_path, index=False, encoding='utf-8-sig')
            
            print("\n" + "*"*50)
            print("SUCCESS: New clean metadata file created!")
            print(f"Original File (Untouched): {original_csv_path} ({initial_len} rows)")
            print(f"New Clean File:            {output_path} ({len(df_clean)} rows)")
            print(f"Removed Rows:              {len(indices_to_drop)}")
            print("*"*50)
            print(f"IMPORTANT: Go to config.py and change METADATA_PATH to point to: '{new_filename}'")
            
        except Exception as e:
            print(f"Error creating clean CSV: {e}")
    else:
        print("\n[INFO] No new file created. Original metadata remains unchanged.")

def denormalize_image(tensor_img):
    img = tensor_img.permute(1, 2, 0).cpu().numpy()
    img = np.clip(img, 0, 1)
    return img

def visualize_sample(ds, index):
    image, mask = ds[index]
    img_display = denormalize_image(image)
    mask_display = mask.cpu().numpy()
    thermal_2d = img_display[:, :, 0]
    
    print(f"Showing sample Index: {index}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(thermal_2d, cmap='inferno') 
    axes[0].set_title(f"Temperature Image [Idx: {index}]")
    axes[0].axis('off')
    
    im_mask = axes[1].imshow(mask_display, cmap='jet', vmin=0, vmax=3, interpolation='nearest')
    axes[1].set_title(f"Mask Labels: {np.unique(mask_display)}")
    axes[1].axis('off')
    
    cbar = plt.colorbar(im_mask, ax=axes[1], ticks=[0, 1, 2, 3], fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels(['Background', 'Head', 'Body', 'Tail'])
    
    axes[2].imshow(thermal_2d, cmap='inferno')
    axes[2].imshow(mask_display, cmap='jet', vmin=0, vmax=3, interpolation='nearest', alpha=0.5)
    axes[2].set_title("Overlay")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
