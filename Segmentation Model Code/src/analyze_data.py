import pandas as pd
import numpy as np
import cv2
import torch
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns # אופציונלי, לגרפים יפים יותר
import config
from dataset import imread_unicode  # טעינת הפונקציה הקיימת שלך

def calculate_class_weights():
    # 1. טעינת ה-Metadata
    csv_path = config.METADATA_PATH
    print(f"--- Loading Metadata from {csv_path} ---")
    
    # בדיקה שהקובץ קיים
    if not os.path.exists(csv_path):
        print(f"[ERROR] Metadata file not found at: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    # אתחול ספירה
    num_classes = config.NUM_CLASSES
    pixel_counts = np.zeros(num_classes, dtype=np.int64)
    class_names = ['Background', 'Head', 'Body', 'Tail']
    
    print("--- Scanning Masks for Pixel Distribution ---")
    # לולאה על כל המסיכות
    valid_masks = 0
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        mask_path = row['mask_abspath']
        
        # שימוש ב-loader שלך להתמודדות עם נתיבים בעברית
        mask = imread_unicode(mask_path, cv2.IMREAD_UNCHANGED)
        
        if mask is None:
            # print(f"[WARNING] Could not load mask at: {mask_path}") # Uncomment if too noisy
            continue
            
        valid_masks += 1
        # ספירת פיקסלים (Flattening למהירות)
        labels, counts = np.unique(mask, return_counts=True)
        for label, count in zip(labels, counts):
            if label < num_classes:
                pixel_counts[label] += count
    
    if valid_masks == 0:
        print("[ERROR] No valid masks were loaded. Check paths in metadata.")
        return

    total_pixels = pixel_counts.sum()
    frequencies = pixel_counts / total_pixels
    
    # יצירת DataFrame לסיכום
    stats_df = pd.DataFrame({
        'class_id': list(range(num_classes)),
        'class_name': class_names,
        'pixel_count': pixel_counts,
        'frequency': frequencies,
        'percent': frequencies * 100
    })
    
    print("\n--- Class Distribution ---")
    print(stats_df)
    
    # --- ויזואליזציה (Histogram) ---
    plt.figure(figsize=(10, 6))
    
    # צבעים לכל מחלקה (רקע אפור, ראש אדום, גוף כחול, זנב ירוק - לדוגמה)
    colors = ['#d3d3d3', '#ff6666', '#66b3ff', '#99ff99']
    
    bars = plt.bar(stats_df['class_name'], stats_df['frequency'], color=colors, edgecolor='black', alpha=0.8)
    
    plt.title(f'Normalized Class Distribution (Total Pixels: {total_pixels:,})', fontsize=15)
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Frequency (0-1)', fontsize=12)
    plt.ylim(0, max(frequencies) * 1.15)  # קצת מרווח למעלה לטקסט
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # הוספת אחוזים מעל העמודות
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                 f'{height:.1%}',
                 ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # שמירת הגרף
    save_path = 'class_distribution_hist.png'
    plt.savefig(save_path, dpi=300)
    print(f"\n[INFO] Histogram saved to: {os.path.abspath(save_path)}")
    plt.show()
    
    # --- חישוב משקולות בשיטות שונות ---
    weights_dict = {}
    
    # 1. Inverse Frequency
    w_inv = 1 / (frequencies + 1e-6)
    weights_dict['inverse_freq'] = w_inv / w_inv.mean() 
    
    # 2. Sqrt Inverse Frequency (מומלץ לרוב)
    w_sqrt = np.sqrt(1 / (frequencies + 1e-6))
    weights_dict['sqrt_inverse'] = w_sqrt / w_sqrt.mean()
    
    # 3. Median Frequency Balancing
    median_freq = np.median(frequencies)
    w_median = median_freq / (frequencies + 1e-6)
    weights_dict['median_freq'] = w_median / w_median.mean()
    
    # 4. Effective Number of Samples (ENS)
    beta = 0.9999
    effective_num = (1.0 - np.power(beta, pixel_counts)) / (1.0 - beta)
    w_ens = (1.0 - beta) / effective_num
    w_ens = w_ens / w_ens.mean()
    weights_dict['effective_num'] = w_ens
    
    # הצגת טבלת השוואה
    print("\n--- Calculated Class Weights (Normalized, Mean=1.0) ---")
    weights_comparison = pd.DataFrame(weights_dict)
    weights_comparison.insert(0, 'class_name', class_names)
    print(weights_comparison)
    
    # בחירה בשיטת ENS + המלצה ידנית (Boost לזנב)
    final_weights = weights_dict['effective_num'].copy()
    
    # המלצה סופית לקוד:
    print(f"\n[RECOMMENDATION] Copy this line to your train.py:")
    print("-" * 60)
    # בחירה בשיטה שעבדה הכי טוב לפי הפלט שלך (Sqrt Inverse)
    # שיטת effective_num נכשלה (החזירה 1.0) בגלל כמות הפיקסלים העצומה
    final_weights = weights_dict['sqrt_inverse'].copy()
    
    # אופציונלי: חיזוק ידני לזנב (Tail - index 3) אם רוצים להיות אגרסיביים יותר
    # final_weights[3] *= 1.5 
    
    # נרמול מחדש סביב 1.0
    final_weights = final_weights / final_weights.mean()
    
    # המרת הנתונים לרשימה נקייה של מספרים (Python floats) להדפסה יפה
    clean_weights_list = [round(float(x), 4) for x in final_weights]
    
    print("\n" + "="*60)
    print(f"[RECOMMENDATION] Copy this line to your train.py:")
    print("="*60)
    # בניית המחרוזת בצורה ברורה
    weights_str = f"[{clean_weights_list[0]}, {clean_weights_list[1]}, {clean_weights_list[2]}, {clean_weights_list[3]}]"
    print(f"# Class mapping: 0=Background, 1=Head, 2=Body, 3=Tail")
    print(f"class_weights = {weights_str}")
    print("="*60)
    
    return stats_df, final_weights

def save_fold_distribution(df_train, df_val, df_test, fold_num):
    """
    Saves a histogram showing 3-way distribution (Train, Val, Test).
    """
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))    

    if str(fold_num).lower() == 'baseline':
        plot_title = "Baseline Split (Train / Val / Test)"
        filename = "baseline_distribution.png"
        SAVE_DIR = os.path.join(BASE_DIR, '..', 'baseline histograms')
    else:
        plot_title = f"Fold {fold_num} Distribution (Train / Val / Test)"
        filename = f"fold_{fold_num}_distribution.png"
        SAVE_DIR = os.path.join(BASE_DIR, '..', 'kfold histograms')
    
        
    os.makedirs(SAVE_DIR, exist_ok=True)
    # Prepare Data
    df_train_temp = df_train[['rat_unique_id']].copy()
    df_train_temp['Set'] = 'Train'
    
    df_val_temp = df_val[['rat_unique_id']].copy()
    df_val_temp['Set'] = 'Validation'
    
    df_test_temp = df_test[['rat_unique_id']].copy()
    df_test_temp['Set'] = 'Test'
    
    combined_df = pd.concat([df_train_temp, df_val_temp, df_test_temp])
    
    # Count occurrences
    data_counts = combined_df.groupby(['rat_unique_id', 'Set']).size().unstack(fill_value=0)
    
    # Sort Numerically
    def extract_rat_num(name):
        try:
            return int(name.replace('Rat', ''))
        except:
            return 999 

    sorted_rats = sorted(data_counts.index.tolist(), key=extract_rat_num)
    data_counts = data_counts.reindex(sorted_rats)

    # Plot
    plt.figure(figsize=(14, 7))
    # Colors: Train=Blue, Val=Orange, Test=Green
    data_counts.plot(kind='bar', stacked=True, color=['#1f77b4', '#ff7f0e', '#2ca02c'], width=0.8, ax=plt.gca())
    
    plt.title(plot_title, fontsize=16)
    plt.xlabel('Rat Unique ID', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Dataset Split')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    save_path = os.path.join(SAVE_DIR, filename)
    plt.savefig(save_path, dpi=150)
    plt.close() 
    print(f"[INFO] Distribution histogram saved to: {save_path}")

if __name__ == "__main__":
    calculate_class_weights()