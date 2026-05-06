import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import config

# --- Helper function for Hebrew/Unicode paths on Windows ---
def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    try:
        # Read file as byte stream into numpy array
        stream = np.fromfile(path, np.uint8)
        # Decode stream to image
        image = cv2.imdecode(stream, flags)
        return image
    except Exception as e:
        print(f"Error loading image at {path}: {e}")
        return None

class RatThermalDataset(Dataset):
    def __init__(self, df, transforms=None):
        self.df = df.reset_index(drop=True)
        self.image_paths = self.df['csv_abspath'].values
        self.mask_paths = self.df['mask_abspath'].values
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # --- 1. Load Mask ---
        mask_path = self.mask_paths[idx]
        
        # Use IMREAD_UNCHANGED to avoid color conversion issues
        mask = imread_unicode(mask_path, cv2.IMREAD_UNCHANGED)
        
        if mask is None:
            print(f"Warning: Mask not found or empty at {mask_path}")
            mask = np.zeros((config.IMG_HEIGHT, config.IMG_WIDTH), dtype=np.uint8)
        
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        # --- 2. Load CSV (Temperature Matrix) ---
        csv_path = self.image_paths[idx]
        try:
            temp_df = pd.read_csv(csv_path, header=None, skiprows=2)
            temp_matrix = temp_df.iloc[:, 1:].values.astype(np.float32)
        except Exception as e:
            print(f"Error loading CSV at {csv_path}: {e}")
            temp_matrix = np.zeros((config.IMG_HEIGHT, config.IMG_WIDTH), dtype=np.float32)

        # --- 3. Rotation Logic ---
        h, w = temp_matrix.shape
        if h < w: 
            temp_matrix = cv2.rotate(temp_matrix, cv2.ROTATE_90_CLOCKWISE)
            mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)

        # --- 4. Normalization ---
        min_val = 20
        max_val = 40
        
        if max_val > min_val:
            image = (temp_matrix - min_val) / (max_val - min_val)
        else:
            image = np.zeros_like(temp_matrix)

        image = np.stack([image, image, image], axis=-1)

        # --- DEBUG CHECK 1: BEFORE TRANSFORMS ---
        # Check unique values right before entering Albumentations
        unique_before = np.unique(mask)
            
        # Alert if we are missing labels before transform
        if len(unique_before) < 4:
             print(f"[WARNING] Sample {idx} loaded with missing labels BEFORE transform: {unique_before}")

        # --- 5. Augmentations ---
        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
            # --- DEBUG CHECK 2: AFTER TRANSFORMS ---
            # Convert to numpy for checking if it's a tensor
            mask_np = mask.numpy() if hasattr(mask, 'numpy') else mask
            unique_after = np.unique(mask_np)
            

        return image, mask.long()

# --- Transforms  ---
def get_train_transforms():
    return A.Compose([
        A.Resize(height=config.IMG_HEIGHT, width=config.IMG_WIDTH, interpolation=cv2.INTER_NEAREST),
        
        A.HorizontalFlip(p=0.5),
        
        # שינוי: שימוש ב-ShiftScaleRotate במקום Affine למניעת ה-Warning
        A.ShiftScaleRotate(
            shift_limit=0.1,    # תואם ל-translate_percent
            scale_limit=0.1,    # תואם ל-scale 0.9-1.1
            rotate_limit=15,    # תואם ל-rotate
            p=0.5, 
            border_mode=cv2.BORDER_CONSTANT, 
            value=0
        ),
        
        ToTensorV2()
    ])

def get_val_transforms():
    return A.Compose([
        A.Resize(height=config.IMG_HEIGHT, width=config.IMG_WIDTH, interpolation=cv2.INTER_NEAREST),
        ToTensorV2()
    ])