"""
Handwriting Emotion Detection Package.

A complete machine learning pipeline for detecting emotional states
from handwriting using the EMOTHAW dataset.
"""

__version__ = "1.0.0"
__author__ = "ML Handwriting Lab"

from .config import (
    TARGET_EMOTION, USE_BINARY, NUM_CLASSES, CLASS_NAMES,
    DATASET_ROOT, DASS_SCORES_PATH, MODEL_DIR, PLOTS_DIR,
    IMAGE_SIZE, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS
)

from .preprocessing.svc_parser import parse_svc, load_all_svc_files
from .data.label_loader import load_labels, load_dass_scores
from .data.dataset import EmothawImageDataset, get_dataloaders
from .features.image_generator import trajectory_to_image
from .features.signal_features import extract_signal_features, extract_batch_features
from .models.cnn_model import EmotionCNN, EmotionResNet, get_model
from .utils.visualization import plot_trajectory, plot_dataset_statistics
from .evaluation.evaluate import evaluate_model, plot_confusion_matrix, plot_training_history
from .inference.predict import load_trained_model, predict

__all__ = [
    "TARGET_EMOTION", "USE_BINARY", "NUM_CLASSES", "CLASS_NAMES",
    "DATASET_ROOT", "DASS_SCORES_PATH", "MODEL_DIR", "PLOTS_DIR",
    "IMAGE_SIZE", "BATCH_SIZE", "LEARNING_RATE", "NUM_EPOCHS",
    "parse_svc", "load_all_svc_files", "load_labels", "load_dass_scores",
    "EmothawImageDataset", "get_dataloaders",
    "trajectory_to_image",
    "extract_signal_features", "extract_batch_features",
    "EmotionCNN", "EmotionResNet", "get_model",
    "plot_trajectory", "plot_dataset_statistics",
    "evaluate_model", "plot_confusion_matrix", "plot_training_history",
    "load_trained_model", "predict",
]
