import torch
import torch.nn as nn
import numpy as np
import cv2
import segmentation_models_pytorch as smp
import config
class RatThermalSegmentor(nn.Module):
    """
    Wrapper class for the U-Net model. 
    It handles the standard forward pass for training, 
    and includes built-in post-processing (blob cleaning) for inference.
    """
    def __init__(self):
        super(RatThermalSegmentor, self).__init__()
        # Build the base U-Net model with ResNet34 backbone
        self.base_model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,             # Replicated to 3 channels as per preprocessing
            classes=4                  # 0: background, 1: head, 2: body, 3: tail
        )

    def forward(self, x):
        """
        Standard forward pass used during training.
        Returns raw logits (B, C, H, W).
        """
        return self.base_model(x)

    def _keep_largest_blob(self, mask_pred):
        """
        Internal helper method: Removes noise using 8-connectivity.
        Keeps only the largest connected component among the non-background classes.
        
        Args:
            mask_pred (np.array): Single mask (H, W) with class indices 0-3.
        Returns:
            cleaned_mask (np.array): The cleaned mask.
        """
        # Create a binary mask: anything that is not background (0) is considered 'Rat'
        binary_mask = (mask_pred > 0).astype(np.uint8)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        
        # If only background is detected, return original
        if num_labels <= 1:
            return mask_pred

        # Find the index of the largest component (ignoring index 0 which is background)
        # stats columns: [left, top, width, height, area]
        largest_label_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        
        # Create a new mask keeping only the pixels belonging to the largest component
        cleaned_mask = np.zeros_like(mask_pred)
        mask_to_keep = (labels == largest_label_idx)
        
        # Restore the original class values (Head/Body/Tail) for the kept component
        cleaned_mask[mask_to_keep] = mask_pred[mask_to_keep]
        
        return cleaned_mask

    def predict_and_clean(self, x):
        """
        Runs inference AND post-processing.
        
        Args:
            x (torch.Tensor): Input batch (B, C, H, W).
            
        Returns:
            np.array: Batch of cleaned masks (B, H, W).
        """
        # Ensure model is in eval mode
        self.eval()
        
        with torch.no_grad():
            # Get raw logits from the base model
            logits = self.forward(x) 
            
            # Convert logits to class indices (B, H, W)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        
        # Apply cleaning to each image in the batch
        cleaned_batch = []
        for i in range(preds.shape[0]):
            cleaned_mask = self._keep_largest_blob(preds[i])
            cleaned_batch.append(cleaned_mask)
            
        return np.array(cleaned_batch)

def build_model():
    """
    Factory function to instantiate the custom wrapper class.
    """
    return RatThermalSegmentor()