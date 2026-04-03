#!/usr/bin/env python3
"""
Main entry point for Handwriting Emotion Detection project.

Provides a unified CLI interface for training, inference, visualization, and analysis.
"""
import os
import sys
import argparse
import torch
import numpy as np

# Ensure the project is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Handwriting Emotion Detection — Complete ML Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py train --model cnn
  python main.py train --model resnet --epochs 20
  python main.py predict --svc_path path/to/handwriting.svc
  python main.py evaluate
  python main.py visualize
  python main.py info
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # ─── TRAIN COMMAND ───────────────────────────────────────────────
    train_parser = subparsers.add_parser("train", help="Train emotion detection model")
    train_parser.add_argument("--model", type=str, default="cnn",
                              choices=["cnn", "resnet"],
                              help="Model architecture (default: cnn)")
    train_parser.add_argument("--epochs", type=int, default=None,
                              help="Override number of training epochs")
    train_parser.add_argument("--batch_size", type=int, default=None,
                              help="Override batch size")
    train_parser.add_argument("--lr", type=float, default=None,
                              help="Override learning rate")
    train_parser.add_argument("--emotion", type=str, default=None,
                              choices=["depression","anxiety","stress"],
                              help="Target emotion for training (overrides config)")

    # batch training for all emotions
    train_all_parser = subparsers.add_parser("train_all",
                                             help="Train a model for each supported emotion")
    train_all_parser.add_argument("--model", type=str, default="cnn",
                                  choices=["cnn", "resnet"],
                                  help="Model architecture to train for all emotions")
    train_all_parser.add_argument("--epochs", type=int, default=None,
                                  help="Override number of training epochs")
    train_all_parser.add_argument("--batch_size", type=int, default=None,
                                  help="Override batch size")
    train_all_parser.add_argument("--lr", type=float, default=None,
                                  help="Override learning rate")

    # ─── PREDICT COMMAND ─────────────────────────────────────────────
    predict_parser = subparsers.add_parser("predict", help="Predict emotion from .svc file")
    predict_parser.add_argument("--svc_path", type=str, required=True,
                                help="Path to .svc handwriting file")
    predict_parser.add_argument("--model_path", type=str, default=None,
                                help="Path to trained model checkpoint")
    predict_parser.add_argument("--emotion", type=str, default=None,
                                choices=["depression","anxiety","stress"],
                                help="Target emotion override when predicting")

    # ─── VISUALIZE COMMAND ───────────────────────────────────────────
    viz_parser = subparsers.add_parser("visualize",
                                       help="Generate visualizations from dataset")
    viz_parser.add_argument("--trajectory", type=str, default=None,
                            help="User ID to visualize trajectory (e.g., 1)")
    viz_parser.add_argument("--statistics", action="store_true",
                            help="Plot dataset class distribution")
    viz_parser.add_argument("--all", action="store_true",
                            help="Generate all visualizations")

    # ─── EVALUATE COMMAND ────────────────────────────────────────────
    eval_parser = subparsers.add_parser("evaluate",
                                        help="Evaluate model on test set and display metrics")

    # ─── EXTRACT COMMAND ─────────────────────────────────────────────
    extract_parser = subparsers.add_parser("extract",
                                           help="Extract signal features from handwriting")
    extract_parser.add_argument("--svc_path", type=str,
                                help="Path to .svc file for feature extraction")
    extract_parser.add_argument("--dataset", action="store_true",
                                help="Extract features for entire dataset")

    # ─── INFO COMMAND ────────────────────────────────────────────────
    info_parser = subparsers.add_parser("info",
                                        help="Display project and dataset information")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "train":
        run_train(args)
    elif args.command == "train_all":
        run_train_all(args)
    elif args.command == "predict":
        run_predict(args)
    elif args.command == "visualize":
        run_visualize(args)
    elif args.command == "evaluate":
        run_evaluate(args)
    elif args.command == "extract":
        run_extract(args)
    elif args.command == "info":
        run_info(args)


def run_train(args):
    """Run training pipeline."""
    from config import NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, TARGET_EMOTION, USE_BINARY, CLASS_NAMES
    from training.train_cnn import train

    print("\n" + "=" * 70)
    print("  HANDWRITING EMOTION DETECTION — TRAINING")
    print("=" * 70)

    # Override config if specified
    if args.epochs:
        import config
        config.NUM_EPOCHS = args.epochs
        print(f"✓ Overriding epochs: {args.epochs}")

    if args.batch_size:
        import config
        config.BATCH_SIZE = args.batch_size
        print(f"✓ Overriding batch size: {args.batch_size}")

    if args.lr:
        import config
        config.LEARNING_RATE = args.lr
        print(f"✓ Overriding learning rate: {args.lr}")

    if args.emotion:
        import config
        config.TARGET_EMOTION = args.emotion
        print(f"✓ Overriding target emotion: {args.emotion}")

    print(f"\nConfiguration:")
    print(f"  Model: {args.model.upper()}")
    print(f"  Target emotion: {TARGET_EMOTION}")
    print(f"  Binary classification: {USE_BINARY}")
    print(f"  Classes: {CLASS_NAMES}")

    train(model_type=args.model)



def run_train_all(args):
    """Train models sequentially for every emotion in config.EMOTIONS."""
    from config import EMOTIONS, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, USE_BINARY, CLASS_NAMES
    from training.train_cnn import train

    print("\n" + "=" * 70)
    print("  HANDWRITING EMOTION DETECTION — BATCH TRAINING")
    print("=" * 70)

    # common overrides
    if args.epochs:
        import config
        config.NUM_EPOCHS = args.epochs
        print(f"✓ Overriding epochs: {args.epochs}")
    if args.batch_size:
        import config
        config.BATCH_SIZE = args.batch_size
        print(f"✓ Overriding batch size: {args.batch_size}")
    if args.lr:
        import config
        config.LEARNING_RATE = args.lr
        print(f"✓ Overriding learning rate: {args.lr}")

    for emotion in EMOTIONS:
        print(f"\n--- training for emotion '{emotion}' ---")
        # set configuration
        import config
        config.TARGET_EMOTION = emotion
        print(f"  target emotion now: {config.TARGET_EMOTION}")
        print(f"  binary classification: {config.USE_BINARY}")
        print(f"  classes: {config.CLASS_NAMES}")
        train(model_type=args.model)
    print("\nBatch training complete. Models saved under MODEL_DIR with emotion tags.")

def run_predict(args):
    """Run inference on a single .svc file."""
    from inference.predict import predict, load_trained_model
    from config import CLASS_NAMES, TARGET_EMOTION

    print("\n" + "=" * 70)
    print("  HANDWRITING EMOTION PREDICTION")
    print("=" * 70)

    if not os.path.exists(args.svc_path):
        print(f"✗ Error: File not found: {args.svc_path}")
        sys.exit(1)

    print(f"\nPredicting emotion from: {args.svc_path}")

    try:
        # pass emotion override to predictor if provided
        kwargs = {}
        if args.emotion:
            kwargs["target_emotion"] = args.emotion
        result = predict(args.svc_path, model_path=args.model_path, **kwargs)
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        sys.exit(1)

    if "error" in result:
        print(f"✗ Error: {result['error']}")
        sys.exit(1)

    print(f"\nInput Statistics:")
    print(f"  Data points: {result['num_data_points']}")
    print(f"  Target dimension: {result['target_emotion'].capitalize()}")
    print(f"  Model: {result.get('model_type', 'unknown').upper()}")

    print(f"\n{'─' * 70}")
    print(f"  PREDICTED EMOTION: {result['predicted_class'].upper()}")
    print(f"  CONFIDENCE: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
    print(f"{'─' * 70}")

    print(f"\nProbability Distribution:")
    for cls_name, prob in result["all_probabilities"].items():
        bar_len = int(prob * 50)
        bar = "█" * bar_len
        print(f"  {cls_name:15s}: {prob:.4f}  {bar}")


def run_visualize(args):
    """Generate visualizations."""
    from preprocessing.svc_parser import parse_svc, load_all_svc_files
    from data.label_loader import load_labels
    from utils.visualization import plot_trajectory, plot_dataset_statistics
    from config import DATASET_ROOT, PLOTS_DIR, TARGET_EMOTION, USE_BINARY, CLASS_NAMES
    import os

    print("\n" + "=" * 70)
    print("  VISUALIZATION GENERATION")
    print("=" * 70)

    os.makedirs(PLOTS_DIR, exist_ok=True)

    if args.trajectory:
        # Plot trajectory for specific user
        user_id = int(args.trajectory)
        samples = load_all_svc_files(DATASET_ROOT)
        user_samples = [s for s in samples if s["user_id"] == user_id]

        if not user_samples:
            print(f"✗ No samples found for user {user_id}")
            return

        sample = user_samples[0]
        save_path = os.path.join(PLOTS_DIR, f"trajectory_user_{user_id:05d}.png")
        plot_trajectory(sample["data"], title=f"User {user_id} — Task {sample['task_id']}",
                        save_path=save_path)
        print(f"✓ Trajectory plot saved: {save_path}")

    if args.statistics or args.all:
        labels = load_labels(target_emotion=TARGET_EMOTION, use_binary=USE_BINARY)
        save_path = os.path.join(PLOTS_DIR, "class_distribution.png")
        plot_dataset_statistics(labels, CLASS_NAMES, TARGET_EMOTION, save_path=save_path)
        print(f"✓ Statistics plot saved: {save_path}")

    if args.all:
        # Generate visualizations for all utility functions
        print(f"✓ Visualization suite generated in: {PLOTS_DIR}")


def run_evaluate(args):
    """Run evaluation on test set."""
    import torch
    from config import MODEL_DIR, TARGET_EMOTION, USE_BINARY, CLASS_NAMES, DATASET_ROOT, DASS_SCORES_PATH, BATCH_SIZE, IMAGE_CACHE_DIR, TRAIN_RATIO, VAL_RATIO, RANDOM_SEED, IMAGE_SIZE
    from preprocessing.svc_parser import load_all_svc_files
    from data.label_loader import load_labels
    from data.dataset import get_dataloaders
    from models.cnn_model import get_model
    from evaluation.evaluate import evaluate_model

    print("\n" + "=" * 70)
    print("  MODEL EVALUATION")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Load model
    model_path = os.path.join(MODEL_DIR, "best_cnn_model.pth")
    if not os.path.exists(model_path):
        print(f"✗ No trained model found at {model_path}")
        print(f"  Please train a model first: python main.py train")
        return

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = get_model(checkpoint.get("model_type", "cnn"), 
                     num_classes=checkpoint.get("num_classes", len(CLASS_NAMES)))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    # Load data
    print("\nLoading dataset...")
    samples = load_all_svc_files(DATASET_ROOT)
    labels = load_labels(DASS_SCORES_PATH, TARGET_EMOTION, USE_BINARY)
    _, _, test_loader = get_dataloaders(
        samples, labels,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        cache_dir=IMAGE_CACHE_DIR,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        random_seed=RANDOM_SEED,
    )

    # Evaluate
    print("\nEvaluating model...")
    results = evaluate_model(model, test_loader, device, class_names=CLASS_NAMES)
    print(f"\n✓ Evaluation complete!")


def run_extract(args):
    """Extract features from handwriting."""
    from preprocessing.svc_parser import parse_svc, load_all_svc_files
    from features.signal_features import extract_signal_features, extract_batch_features
    from config import DATASET_ROOT

    print("\n" + "=" * 70)
    print("  FEATURE EXTRACTION")
    print("=" * 70)

    if args.svc_path:
        if not os.path.exists(args.svc_path):
            print(f"✗ File not found: {args.svc_path}")
            return

        print(f"\nExtracting features from: {args.svc_path}")
        data = parse_svc(args.svc_path)
        features = extract_signal_features(data)

        print(f"\nExtracted {len(features)} signal-based features:")
        print("─" * 70)
        for name, value in features.items():
            print(f"  {name:30s}: {value:12.4f}")

    elif args.dataset:
        print("\nExtracting features for entire dataset...")
        samples = load_all_svc_files(DATASET_ROOT)
        feature_matrix, feature_names = extract_batch_features(samples)

        print(f"\n✓ Extracted {len(feature_names)} features from {len(samples)} samples")
        print(f"  Feature matrix shape: {feature_matrix.shape}")
        print(f"  Sample features: {feature_names[:5]}...")

    else:
        print("Please specify --svc_path or --dataset flag")


def run_info(args):
    """Display project and dataset information."""
    from config import (
        DATASET_ROOT, DASS_SCORES_PATH, TARGET_EMOTION, USE_BINARY,
        NUM_CLASSES, CLASS_NAMES, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS
    )
    from preprocessing.svc_parser import load_all_svc_files
    from data.label_loader import load_labels
    import xlrd

    print("\n" + "=" * 70)
    print("  PROJECT & DATASET INFORMATION")
    print("=" * 70)

    print("\n📋 PROJECT CONFIGURATION")
    print("─" * 70)
    print(f"  Target Emotion: {TARGET_EMOTION.upper()}")
    print(f"  Classification Mode: {'Binary (Low/High)' if USE_BINARY else 'Multi-Class (5 levels)'}")
    print(f"  Number of Classes: {NUM_CLASSES}")
    print(f"  Class Names: {', '.join(CLASS_NAMES)}")

    print("\n⚙️  TRAINING CONFIGURATION")
    print("─" * 70)
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Default Epochs: {NUM_EPOCHS}")

    print("\n📁 DATASET INFORMATION")
    print("─" * 70)
    if os.path.exists(DATASET_ROOT):
        samples = load_all_svc_files(DATASET_ROOT)
        labels = load_labels(target_emotion=TARGET_EMOTION, use_binary=USE_BINARY)

        num_users = len(labels)
        num_samples = len(samples)
        num_users_with_data = len(set(s["user_id"] for s in samples if s["user_id"] in labels))

        print(f"  Total users in dataset: {num_users}")
        print(f"  Users with handwriting data: {num_users_with_data}")
        print(f"  Total handwriting samples: {num_samples}")
        print(f"  Tasks per user: up to 7 (different writing tasks)")

        print("\n  Class Distribution:")
        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            count = sum(1 for v in labels.values() if v == cls_idx)
            print(f"    {cls_name}: {count} users")

        # Dataset statistics
        print("\n  Sample Statistics:")
        num_points = [s["data"].shape[0] for s in samples]
        print(f"    Min points: {min(num_points)}")
        print(f"    Max points: {max(num_points)}")
        print(f"    Mean points: {np.mean(num_points):.1f}")
    else:
        print(f"  ⚠️  Dataset not found at {DATASET_ROOT}")

    print("\n🎯 MODEL ARCHITECTURES")
    print("─" * 70)
    print("  1. EmotionCNN")
    print("     - Lightweight custom CNN (4 conv blocks)")
    print("     - ~2.7M parameters")
    print("     - Recommended for small datasets")
    print("\n  2. EmotionResNet")
    print("     - ResNet18 with transfer learning")
    print("     - Fine-tuned on EMOTHAW data")
    print("     - ~11.2M parameters")

    print("\n📚 FEATURES")
    print("─" * 70)
    print("  Image-Based (Default):")
    print("    - Pen trajectories rendered to 224×224 grayscale images")
    print("    - CNN learns spatial patterns of handwriting")
    print("\n  Signal-Based (Optional):")
    print("    - 30+ handwriting signal features")
    print("    - Velocity, pressure, curvature, pen lifts, etc.")
    print("    - Extractable via: python -m handwriting_emotion_detection extract")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
