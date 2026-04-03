"""
Central configuration for the Handwriting Emotion Detection project.
Paths, hyperparameters, and DASS-21 thresholds.
"""
import os

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(
    os.path.dirname(BASE_DIR), "archive", "DataEmothaw"
)
DASS_SCORES_PATH = os.path.join(DATASET_ROOT, "DASS_scores.xls")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
IMAGE_CACHE_DIR = os.path.join(OUTPUT_DIR, "image_cache")

# Create output directories
for _d in [OUTPUT_DIR, MODEL_DIR, PLOTS_DIR, IMAGE_CACHE_DIR]:
    os.makedirs(_d, exist_ok=True)

# ─── SVC File Columns ────────────────────────────────────────────────────────
SVC_COLUMNS = ["x", "y", "timestamp", "pen_status", "azimuth", "altitude", "pressure"]

# ─── DASS-21 Severity Thresholds ─────────────────────────────────────────────
# Standard DASS-21 cutoffs (scores are already * 2 in EMOTHAW)
# Each maps to: Normal, Mild, Moderate, Severe, Extremely Severe
DASS_THRESHOLDS = {
    "depression": [0, 10, 14, 21, 28],   # Normal < 10, Mild 10-13, Moderate 14-20, Severe 21-27, ExSevere >= 28
    "anxiety":    [0, 8, 10, 15, 20],     # Normal < 8,  Mild 8-9,   Moderate 10-14, Severe 15-19, ExSevere >= 20
    "stress":     [0, 15, 19, 26, 34],    # Normal < 15, Mild 15-18, Moderate 19-25, Severe 26-33, ExSevere >= 34
}

DASS_SEVERITY_LABELS = ["Normal", "Mild", "Moderate", "Severe", "Extremely Severe"]

# For binary classification (simpler, better performance on small dataset)
# Normal+Mild = 0 ("Low"), Moderate+Severe+ExSevere = 1 ("High")
BINARY_THRESHOLD = {
    "depression": 14,
    "anxiety": 10,
    "stress": 19,
}

# ─── Target Emotion ──────────────────────────────────────────────────────────
# Which DASS dimension to classify: "depression", "anxiety", or "stress"
TARGET_EMOTION = "anxiety"

# list of all supported emotions – useful for UI and scripts
EMOTIONS = ["depression", "anxiety", "stress"]

# Whether to use binary ("Low"/"High") or multi-class severity labels
USE_BINARY = True

# ─── Image Generation ────────────────────────────────────────────────────────
IMAGE_SIZE = 224          # Rendered trajectory image dimension (224x224)
LINE_THICKNESS = 2        # Base line thickness for drawing strokes

# ─── Model & Training Hyperparameters ────────────────────────────────────────
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 7
WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 0.5

# Train/Val/Test split ratios (by user, to avoid data leakage)
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Random seed for reproducibility
RANDOM_SEED = 42

# ─── Number of classes (auto-set based on USE_BINARY) ────────────────────────
NUM_CLASSES = 2 if USE_BINARY else len(DASS_SEVERITY_LABELS)
CLASS_NAMES = ["Low", "High"] if USE_BINARY else DASS_SEVERITY_LABELS
