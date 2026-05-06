# 🐀 Thermal Image Segmentation of Rats for Anxiety Level Classification

> **Final Project** | Deep Learning-based semantic segmentation of rats in thermal imagery for PTSD/anxiety-level research.

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Dataset Description](#-dataset-description)
3. [Methodology](#-methodology)
4. [Technical Details](#-technical-details)
5. [Training Process](#-training-process)
6. [Results](#-results)
7. [How to Run](#-how-to-run)
8. [Project Structure](#-project-structure)
9. [Requirements](#-requirements)

---

## 🧠 Project Overview

Anxiety and PTSD research in animal models often relies on behavioral and physiological indicators captured via **thermal imaging**. This project develops an automated deep learning pipeline to **segment rats in thermal images** into anatomical body regions — enabling downstream analysis of temperature distribution patterns linked to anxiety levels.

The pipeline:
- Ingests raw **thermal CSV data** (temperature matrices) from a FLIR ONE camera.
- Produces **pixel-level segmentation masks** classifying each pixel as one of four classes: **Background**, **Head**, **Body**, or **Tail**.
- Is evaluated using **5-Fold Nested Cross-Validation** to ensure robust, generalizable performance across different rats.

| Class | Label | Description |
|-------|-------|-------------|
| 0 | Background | Non-rat pixels |
| 1 | Head | Rat head region |
| 2 | Body | Rat torso region |
| 3 | Tail | Rat tail region |

---

## 📦 Dataset Description

### Source
- **25 rats** were recorded using a thermal camera, producing temperature matrices stored as **CSV files**.
- Each CSV encodes pixel-level temperatures (°C) across a 2D spatial grid.

### Dataset Size
- The labeled dataset has grown from the original 435 images to a total of **1,649 images** (435 original + 1,214 manually refined and added). All reported results use the expanded dataset.

### Ground Truth Annotation
- Segmentation masks were **manually annotated** using the **MATLAB Medical Image Labeler** toolbox.
- Each mask is a grayscale PNG where pixel values correspond to class labels (0–3).
- Masks and images are stored per-rat under `Segmentation model/Data For Segmentation/Rat{N}/`. 

### Data Split
| Split | Strategy | Rats |
|-------|----------|------|
| Baseline | 60% Train / 20% Val / 20% Test (fixed) | Test: Rat1, Rat2, Rat8, Rat17, Rat24 |
| K-Fold CV | 5-Fold Nested (Group-based, rat-level) | 5 × ~60/20/20 splits |

> All splits are performed at the **rat level** (group-aware) to prevent data leakage between subjects.

### Preprocessing

1. **Load** temperature CSV → 2D `float32` matrix.
2. **Rotate** landscape images to portrait orientation.
3. **Normalize** temperatures to `[0, 1]` using a fixed thermal range of **20–40 °C**:

```python
image = (temp_matrix - 20.0) / (40.0 - 20.0)
```

4. **Replicate** single-channel to 3-channel tensor (for ImageNet-pretrained encoder compatibility).

### Class Distribution (Pixel-level)

The dataset is imbalanced — background pixels dominate. Weighted loss is used to compensate.

| Class | Weight |
|-------|--------|
| Background | 0.1984 |
| Head | 1.2151 |
| Body | 0.7004 |
| Tail | **1.8861** |

<p align="center">
  <img src="Segmentation model/Data For Segmentation/Pixel_Labels_Distribution.png" alt="Pixel Label Distribution" width="600"/>
</p>

---

## 🔬 Methodology

### 1. Ground Truth Labeling — MATLAB Medical Image Labeler

Thermal CSV files were first converted to PNG images using MATLAB (`scripts/CSV_TO_PNG.mlx`), then imported into the **MATLAB Medical Image Labeler** app for polygon-based annotation of four anatomical classes.

```
scripts/
├── CSV_TO_PNG.mlx          # Converts thermal CSVs to PNG for labeling
├── build_metadata_table.m  # Builds metadata CSV of image/mask pairs
└── statistics_extraction.py
```

### 2. Quality Assurance

A QA pipeline validates mask integrity before training:
- Checks for missing or corrupt files.
- Verifies label range (0–3).
- Produces a clean `metadata_after_QA.csv` used for all training runs.

### 3. Segmentation Model

A **U-Net** architecture with a **ResNet34 encoder** (pretrained on ImageNet) is used for multi-class semantic segmentation. The model is wrapped in a custom class `RatThermalSegmentor` that adds inference-time post-processing.

### 4. Two-Stage Transfer Learning

| Stage | Encoder | Learning Rate | Max Epochs |
|-------|---------|---------------|------------|
| Stage 1 | **Frozen** | `1e-3` | 20 |
| Stage 2 | **Fine-tuned** | `1e-4` | 15 |

Early stopping (patience = 5) is applied in both stages based on validation loss.

### 5. Post-Processing — Largest Blob Cleaning

At inference time, a connected-component analysis (8-connectivity) removes small spurious predictions, keeping only the **largest connected foreground component**:

```python
binary_mask = (mask_pred > 0).astype(np.uint8)
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
largest_label_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
cleaned_mask[labels == largest_label_idx] = mask_pred[labels == largest_label_idx]
```

### 6. Data Augmentation (Training Only)

```python
A.HorizontalFlip(p=0.5),
A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5)
```

---

## ⚙️ Technical Details

### Model Architecture

```
RatThermalSegmentor
└── U-Net (segmentation_models_pytorch)
    ├── Encoder: ResNet34 (pretrained on ImageNet)
    ├── Input:   (B, 3, 640, 480)
    ├── Output:  (B, 4, 640, 480)  ← raw logits
    └── Classes: 4 (Background, Head, Body, Tail)
```

### Loss Function

A **Weighted Combined Dice + Focal Loss** (50/50):

$$\mathcal{L} = 0.5 \cdot \mathcal{L}_{\text{Dice}} + 0.5 \cdot \mathcal{L}_{\text{Focal}}$$

Both losses use class-level weights to address the class imbalance.

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Image size | 640 × 480 |
| Batch size | 8 |
| Optimizer | Adam |
| LR Stage 1 | `1e-3` |
| LR Stage 2 | `1e-4` |
| Epochs Stage 1 | 20 |
| Epochs Stage 2 | 15 |
| Early stopping patience | 5 |
| K-Folds | 5 |

### Evaluation Metrics

- **mIoU** (mean Intersection over Union, macro-averaged over 4 classes)
- **Dice Coefficient** (per class and averaged)
- Per-class IoU: Background, Head, Body, Tail

```python
# IoU computation (via segmentation_models_pytorch)
tp, fp, fn, tn = get_stats(preds, targets, mode='multiclass', num_classes=4)
miou = iou_score(tp, fp, fn, tn, reduction="macro")

# Dice = 2 * IoU / (1 + IoU)
```

---

## 🏋️ Training Process

The model was trained using a **two-stage transfer learning** strategy. Each stage produces training loss curves, validation loss curves, and per-class mIoU curves that are automatically saved to `training graphs/`.

### Baseline Training

<p align="center">
  <img src="Segmentation%20model/Segmentation%20Model%20Code/training%20graphs/training_graph_baseline.png"
       alt="Baseline Training Graph" width="800"/>
</p>

### K-Fold Cross-Validation Training Curves

<table>
  <tr>
    <td align="center"><b>Fold 1</b></td>
    <td align="center"><b>Fold 2</b></td>
  </tr>
  <tr>
    <td><img src="Segmentation%20model/Segmentation%20Model%20Code/training%20graphs/training_graph_fold_1.png" alt="Fold 1 Training" width="420"/></td>
    <td><img src="Segmentation%20model/Segmentation%20Model%20Code/training%20graphs/training_graph_fold_2.png" alt="Fold 2 Training" width="420"/></td>
  </tr>
  <tr>
    <td align="center"><b>Fold 3</b></td>
    <td align="center"><b>Fold 4</b></td>
  </tr>
  <tr>
    <td><img src="Segmentation%20model/Segmentation%20Model%20Code/training%20graphs/training_graph_fold_3.png" alt="Fold 3 Training" width="420"/></td>
    <td><img src="Segmentation%20model/Segmentation%20Model%20Code/training%20graphs/training_graph_fold_4.png" alt="Fold 4 Training" width="420"/></td>
  </tr>
  <tr>
    <td align="center" colspan="2"><b>Fold 5</b></td>
  </tr>
  <tr>
    <td colspan="2" align="center"><img src="Segmentation%20model/Segmentation%20Model%20Code/training%20graphs/training_graph_fold_5.png" alt="Fold 5 Training" width="420"/></td>
  </tr>
</table>

> Each graph shows **batch training loss**, **validation loss**, **mean IoU (mIoU)**, and **per-class IoU** (Background, Head, Body, Tail) over all epochs across both training stages. The vertical dashed line separates Stage 1 (frozen encoder) from Stage 2 (full fine-tuning).

---

## 📊 Results

### Test Metrics

#### Baseline (Fixed Split)

| Model | Test mIoU |
|-------|-----------|
| Baseline | 0.9066 |

#### K-Fold Cross-Validation — Test mIoU

| Fold | Test mIoU |
|------|-----------|
| Fold 1 | 0.9050 |
| Fold 2 | 0.9107 |
| Fold 3 | 0.9019 |
| Fold 4 | 0.9062 |
| Fold 5 | 0.8967 |
| **Mean** | **0.9041** |

Notes: Baseline and K-Fold values are sourced from `metrics/test/baseline/test_metrics_baseline.json` and the per-fold JSON files in `metrics/test/kfolds/`.

### Per-Class Validation IoU — Baseline (Test Set)

| Class | IoU (Baseline) |
|-------|----------------|
| Background | 0.9935 |
| Head       | 0.9039 |
| Body       | 0.9502 |
| Tail       | 0.7788 |

### Per-Class Validation IoU — K-Fold Averages (Test Sets)

| Class | IoU (K-Fold Average) |
|-------|----------------------|
| Background | 0.9937 |
| Head       | 0.9074 |
| Body       | 0.9530 |
| Tail       | 0.7623 |

> Tail remains the most complex and challenging class due to its small, elongated morphology. However, IoU for the Tail has improved substantially (~13 percentage points) compared to earlier experiments (from ~0.66 to ~0.778), attributable to the larger, higher-quality annotated dataset and careful manual correction of masks.

> Summary of Findings: The retrained model demonstrates exceptional robustness across anatomical regions — overall mIoU increased to ~0.907 (baseline) and K-Fold average to ~0.904. Improvements are most notable for the Tail class, while Head and Body performance also show measurable gains.

### 🖼️ Visual Prediction Example — Rat 1

The figure below shows a full test-set prediction for **Rat 1**, comparing the thermal input, the manually annotated ground truth mask, and the model's cleaned prediction (with blob post-processing applied).

<p align="center">
  <img src="Segmentation%20model/Segmentation%20Model%20Code/test%20preds%20figures/baseline/Rat%201%20Thermal%2021.png"
       alt="Rat 1 Test Prediction — Thermal Input, Ground Truth, Model Output" width="900"/>
</p>

| Panel | Description |
|-------|-------------|
| **Thermal Image (Input)** | Raw grayscale thermal frame fed into the model |
| **Ground Truth Mask** | Manually labelled segmentation (MATLAB Medical Image Labeler) |
| **Model Prediction (Cleaned)** | U-Net output after largest-blob post-processing |
| **Overlay: Thermal + GT** | Ground truth superimposed on the thermal image |
| **Overlay: Thermal + Prediction** | Model prediction superimposed on the thermal image |

> 📌 *Run option `5` in the main menu to generate test figures for all samples automatically.*

---

## 🖥️ Interactive Menu Reference

When you run `python main.py`, the following menu is presented:

```
==================================================
RAT THERMAL SEGMENTATION - PROJECT MENU
==================================================
1.  Run Data QA and Cleaning
2.  Visualize Image Distribution
3.  Run BASELINE Training
4.  Run K-FOLD Training
5.  Run Test Evaluation (Metrics + Figures)
6.  Run Test Visualization (Random Sample)
7.  Run Single Inference & Visualize
8.  RUN FULL PIPELINE (Train All -> Test All)
9.  Print Model Architecture
10. Exit
==================================================
```

| Option | Action | Details |
|--------|--------|---------|
| **1** | Data QA & Cleaning | Validates all image/mask pairs, checks label ranges (0–3), removes corrupt entries, and saves a clean `metadata_after_QA.csv`. **Run this first.** |
| **2** | Visualize Distribution | Plots per-rat image count histograms and class distributions to inspect dataset balance. |
| **3** | Baseline Training | Trains a single model with a fixed 60/20/20 split. Saves the best model to `models/baseline_train/model_baseline_best.pth`. |
| **4** | K-Fold Training | Runs 5-fold nested cross-validation (rat-level groups). Saves one best model per fold to `models/kfolds_train/`. |
| **5** | Test Evaluation | Loads a saved model (baseline or a specific fold), computes mIoU & per-class IoU on the held-out test set, and saves prediction figure panels to `test preds figures/`. |
| **6** | Test Visualization | Loads a saved model and displays a random test-set prediction (thermal input, ground truth, cleaned prediction, overlays). |
| **7** | Single Inference | Runs inference on a single image from a chosen model and fold, then visualizes the result. |
| **8** | Full Pipeline | Sequentially runs Baseline Training → Baseline Test Evaluation → K-Fold Training → K-Fold Test Evaluation for all 5 folds. |
| **9** | Print Architecture | Prints the full `RatThermalSegmentor` model summary to the console. |
| **10** | Exit | Exits the program. |

> **Options 5 & 6** prompt you to choose between `baseline` and `kfolds` mode. For `kfolds`, you are also asked to enter a fold number (1–5).

---

## 🚀 How to Run

### Prerequisites

```bash
pip install torch torchvision
pip install segmentation-models-pytorch
pip install albumentations
pip install opencv-python
pip install pandas numpy matplotlib tqdm scikit-learn
```

### 1. Prepare Data

Ensure the following structure exists:

```
Segmentation model/Data For Segmentation/
├── Rat1/
│   ├── *.csv        ← thermal temperature matrices
│   └── *.png        ← corresponding segmentation masks
├── Rat2/
...
└── Rat25/
```

Run QA to validate and build the metadata table:

```bash
cd "Segmentation model/Segmentation Model Code/src"
python main.py
# → Select option 1: Run Data QA and Cleaning
```

### 2. Train the Model

#### Baseline (Single Split)

```bash
python main.py
# → Select option 3: Run BASELINE Training
```

#### K-Fold Cross-Validation

```bash
python main.py
# → Select option 4: Run K-FOLD Training
```

#### Full Pipeline (Train All → Test All)

```bash
python main.py
# → Select option 8: RUN FULL PIPELINE
```

### 3. Evaluate on Test Set

```bash
python main.py
# → Select option 5: Run Test Evaluation (Metrics + Figures)
# → Choose model type: baseline  OR  kfolds
# → (For kfolds) Enter fold number: 1-5
```

### 4. Run Single Inference

```bash
python main.py
# → Select option 7: Run Single Inference & Visualize
# → Enter model type and fold number
```

### 5. Visualize Data Distribution

```bash
python main.py
# → Select option 2: Visualize Image Distribution
```

---

## 📁 Project Structure

```
ProjectPTSDRats/
├── README.md
├── scripts/
│   ├── CSV_TO_PNG.mlx              # MATLAB: thermal CSV → PNG
│   ├── build_metadata_table.m      # MATLAB: build image-mask metadata
│   └── statistics_extraction.py
│
└── Segmentation model/
    ├── Data For Segmentation/
    │   ├── Rat1/ ... Rat25/        # Per-rat CSVs + masks
    │   ├── metadata_after_QA.csv
    │   └── Pixel_Labels_Distribution.png
    │
    └── Segmentation Model Code/
        ├── metadata_after_QA.csv
        ├── models/
        │   ├── baseline_train/     # Saved baseline model (.pth)
        │   └── kfolds_train/       # Saved per-fold models (.pth)
        ├── metrics/
        │   └── metrics_kfolds.csv  # Training + test metrics per fold
        ├── training graphs/        # Loss / mIoU training curves
        ├── test preds figures/     # Saved test prediction visualizations
        └── src/
            ├── main.py             # Entry point — interactive menu
            ├── config.py           # Hyperparameters and paths
            ├── model.py            # RatThermalSegmentor (U-Net + ResNet34)
            ├── dataset.py          # RatThermalDataset + augmentations
            ├── train.py            # Training loops (baseline & k-fold)
            ├── test.py             # Evaluation and visualization
            ├── utils.py            # Loss function, metrics, plotting
            ├── qa_utils.py         # Data quality assurance
            └── analyze_data.py     # Distribution analysis utilities
```

---

## 📦 Requirements

```
torch>=1.12.0
torchvision
segmentation-models-pytorch
albumentations
opencv-python
pandas
numpy
matplotlib
tqdm
scikit-learn
```

---

## 📄 License

See [LICENSE](LICENSE) for details.

---

<p align="center">
  Made with ❤️ for PTSD rat anxiety research<br/>
  <em>Final Project — Biomedical Engineering / Computer Science</em>
</p>
