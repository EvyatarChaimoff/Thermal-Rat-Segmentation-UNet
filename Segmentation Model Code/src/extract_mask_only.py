import torch
import matplotlib.pyplot as plt
import os
import numpy as np

# ייבוא הפונקציות מהקבצים שלך
import config
from test import load_best_model, preprocess_single_csv

def extract_rat_mask():
    # --- הגדרות ---
    model_type = 'baseline'  # או 'kfolds'
    fold = 1                # רלוונטי רק אם בחרת kfolds
    
    # נתיבים (שימוש ב-r כדי למנוע שגיאות escape sequence)
    csv_path = r"..\..\Data For Segmentation\Rat1\CSV\Thermal_38_CSV.csv"
    output_folder = r"..\..\Mask Comparation\Segementation model predicted masks"
    output_filename = "Rat_1_Thermal_38.png"
    
    # וודא שהתיקייה קיימת
    os.makedirs(output_folder, exist_ok=True)
    full_save_path = os.path.join(output_folder, output_filename)

    # 1. טעינת המודל (מתוך test.py)
    model_instance = load_best_model(model_type, fold=fold if model_type == 'kfolds' else None)
    if model_instance is None:
        return

    # 2. הכנת התמונה לאינפרנס (מתוך test.py)
    # הפונקציה מבצעת סיבוב, נרמול והפיכה ל-Tensor
    image_tensor, _ = preprocess_single_csv(csv_path)
    
    device = config.DEVICE
    model_instance.to(device)
    model_instance.eval()

    # הוספת Batch dimension והעברה ל-GPU
    image_input = image_tensor.unsqueeze(0).to(device, dtype=torch.float32) 

    # 3. הרצת המודל
    with torch.no_grad():
        output = model_instance(image_input) # Output shape: [1, 4, H, W]
        # הוצאת המחלקה עם ההסתברות הגבוהה ביותר לכל פיקסל
        mask_pred = torch.argmax(output.squeeze(0), dim=0).cpu().numpy()
    
    # 4. יצירת מסיכה בינארית (איחוד ROI)
    # כל מה שגדול מ-0 (כלומר מחלקות 1, 2, 3) הופך ל-255 (לבן)
    # כל מה ששווה ל-0 (רקע) נשאר 0 (שחור)
    binary_mask = np.where(mask_pred > 0, 255, 0).astype(np.uint8)

    # 5. שמירה
    plt.imsave(full_save_path, binary_mask, cmap='gray')
    print(f"Success! Binary mask saved to: {full_save_path}")

if __name__ == "__main__":
    extract_rat_mask()