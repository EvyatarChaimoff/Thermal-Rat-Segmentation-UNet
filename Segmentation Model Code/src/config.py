import torch

# --- Paths ---
# Assuming running from 'src' folder
DATA_DIR = '../Data For Segmentation'
METADATA_PATH = '../Data For Segmentation/metadata.csv'
TRAINING_GRAPHS_DIR = '../training graphs'
# --- Hyperparameters ---
# --- Hyperparameters ---
IMG_HEIGHT = 640
IMG_WIDTH = 480
BATCH_SIZE = 8 # Adjust based on your GPU VRAM
NUM_CLASSES = 4 # 0: Background, 1: Head, 2: Body, 3: Tail

# --- Training Config ---
K_FOLDS = 5                 # Cross Validation
TEST_RATS = ['Rat5', 'Rat6', 'Rat7'] # Test
LEARNING_RATE_STAGE1 = 1e-3 # Decoder
LEARNING_RATE_STAGE2 = 1e-4 # Fine tuning
EPOCHS_STAGE1 = 20
EPOCHS_STAGE2 = 15
PATIENCE = 5 # Early Stopping patience

# --- Class Weights (Recommended after analysis) ---
BACKGROUND_WEIGHT = 0.2186
HEAD_WEIGHT = 1.1523
BODY_WEIGHT = 0.6859
TAIL_WEIGHT = 1.9432


# --- Compute ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0 # Set to 0 for debugging on Windows, increase later if stable

BASELINE_MODEL_DIR = "../models/baseline_train/model_baseline_best.pth"