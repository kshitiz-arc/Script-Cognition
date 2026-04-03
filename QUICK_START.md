#!/usr/bin/env python3
"""
QUICK REFERENCE GUIDE — Handwriting Emotion Detection
Copy and paste these commands to quickly get started!
"""

# ═══════════════════════════════════════════════════════════════════════════
# SETUP & VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════

# Navigate to project directory
cd handwriting_emotion_detection

# Verify project is ready
python test_project.py

# View project information
python main.py info


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING MODELS
# ═══════════════════════════════════════════════════════════════════════════

# Train lightweight CNN model (RECOMMENDED)
python main.py train --model cnn

# Train with more epochs
python main.py train --model cnn --epochs 50

# Train ResNet18 (more powerful, slower)
python main.py train --model resnet

# Train ResNet with custom parameters
python main.py train --model resnet --epochs 40 --batch_size 32 --lr 0.0001


# ═══════════════════════════════════════════════════════════════════════════
# MAKING PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════════

# Predict from a single .svc file
python main.py predict --svc_path "path/to/handwriting.svc"

# Predict with specific model
python main.py predict --svc_path "file.svc" --model_path "outputs/models/best_cnn_model.pth"


# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZATION & ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

# View dataset statistics
python main.py visualize --statistics

# Plot specific user's handwriting trajectory
python main.py visualize --trajectory 1     # User ID 1
python main.py visualize --trajectory 5     # User ID 5

# Generate all visualizations
python main.py visualize --all

# View plots in: outputs/plots/


# ═══════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

# Extract signal features from single file
python main.py extract --svc_path "file.svc"

# Extract features for entire dataset
python main.py extract --dataset


# ═══════════════════════════════════════════════════════════════════════════
# EVALUATE TRAINED MODEL
# ═══════════════════════════════════════════════════════════════════════════

# Get test set metrics and confusion matrix
python main.py evaluate


# ═══════════════════════════════════════════════════════════════════════════
# PYTHON API USAGE
# ═══════════════════════════════════════════════════════════════════════════

"""
from handwriting_emotion_detection import (
    load_all_svc_files, load_labels, get_dataloaders,
    EmotionCNN, extract_signal_features, plot_trajectory
)
from config import DATASET_ROOT, DASS_SCORES_PATH

# Load dataset
samples = load_all_svc_files(DATASET_ROOT)
labels = load_labels(DASS_SCORES_PATH, target_emotion="anxiety")

# Create dataloaders
train_loader, val_loader, test_loader = get_dataloaders(
    samples, labels, batch_size=16, image_size=224
)

# Create model
model = EmotionCNN(num_classes=2)

# Extract features from sample
features = extract_signal_features(samples[0]["data"])
print(f"{len(features)} features extracted")

# Visualize trajectory
plot_trajectory(samples[0]["data"], save_path="plot.png")
"""


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION CUSTOMIZATION
# ═══════════════════════════════════════════════════════════════════════════

"""
Edit config.py to customize:

TARGET_EMOTION = "anxiety"          # anxiety, depression, stress
USE_BINARY = True                   # True = binary, False = 5-class
NUM_EPOCHS = 30                     # Training epochs
BATCH_SIZE = 16                     # Batch size
LEARNING_RATE = 1e-4               # Learning rate
IMAGE_SIZE = 224                    # Image size for CNN
EARLY_STOPPING_PATIENCE = 7         # Early stopping patience

Then retrain: python main.py train --model cnn
"""


# ═══════════════════════════════════════════════════════════════════════════
# TROUBLESHOOTING
# ═══════════════════════════════════════════════════════════════════════════

# Check what's installed
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"

# Verify GPU availability
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}')"

# Debug dataset loading
python main.py info

# Check outputs
ls outputs/models/          # View trained models
ls outputs/plots/           # View generated plots
ls outputs/image_cache/     # View cached images


# ═══════════════════════════════════════════════════════════════════════════
# ADVANCED USAGE
# ═══════════════════════════════════════════════════════════════════════════

# Compare two model architectures
echo "Training CNN..."
python main.py train --model cnn --epochs 20

echo "Training ResNet..."
python main.py train --model resnet --epochs 20

# Then evaluate both with:
python main.py evaluate

# Experiment with different emotions
# 1. Edit config.py: TARGET_EMOTION = "stress"
# 2. Train: python main.py train --model cnn
# 3. Compare results


# ═══════════════════════════════════════════════════════════════════════════
# COMMON COMMANDS QUICK REFERENCE
# ═══════════════════════════════════════════════════════════════════════════

python main.py --help                    # See all commands
python main.py train --help              # See training options
python main.py predict --help            # See prediction options
python main.py visualize --help          # See visualization options

python main.py info                      # Project & dataset info
python main.py train --model cnn         # Quick start training
python main.py visualize --statistics    # View data distribution
python test_project.py                   # Verify installation


# ═══════════════════════════════════════════════════════════════════════════
# EXPECTED OUTPUT EXAMPLES
# ═══════════════════════════════════════════════════════════════════════════

"""
Training Output:
  Epoch  1/30 | Train Loss: 0.6829  Acc: 0.5294 | Val Loss: 0.6721  Acc: 0.5500
  Epoch  2/30 | Train Loss: 0.6241  Acc: 0.6176 | Val Loss: 0.6089  Acc: 0.6500
  ...
  ✓ Saved best model (val_loss: 0.5234)

Prediction Output:
  PREDICTED EMOTION: LOW
  CONFIDENCE: 0.8412 (84.12%)
  
  Probability Distribution:
    Low : 0.8412  ████████████████████████████
    High: 0.1588  ████

Test Metrics:
  Accuracy: 0.7412
  Precision: 0.7235
  Recall: 0.7412
  F1-Score: 0.7321
"""


# ═══════════════════════════════════════════════════════════════════════════
# PROJECT FILES OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════

"""
config.py                    ← Edit here to customize parameters
main.py                      ← Run: python main.py <command>
README.md                    ← Full documentation
PROJECT_COMPLETE.md          ← Project completion summary
test_project.py              ← Verify installation

preprocessing/svc_parser.py  ← Parse .svc files
data/label_loader.py         ← Load DASS scores
data/dataset.py              ← PyTorch Dataset/DataLoader
features/image_generator.py  ← Render images from trajectories
features/signal_features.py  ← Extract handwriting features
models/cnn_model.py          ← Model architectures (CNN, ResNet)
training/train_cnn.py        ← Training pipeline
evaluation/evaluate.py       ← Metrics and visualization
inference/predict.py         ← Make predictions
utils/visualization.py       ← Plotting functions

outputs/models/              ← Trained model checkpoints
outputs/plots/               ← Generated visualizations
outputs/image_cache/         ← Cached trajectory images
"""
