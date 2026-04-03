"""
CNN Training Pipeline for Handwriting Emotion Detection.

Complete training loop with early stopping, learning rate scheduling,
and model checkpointing.
"""
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    DATASET_ROOT, DASS_SCORES_PATH, OUTPUT_DIR, MODEL_DIR, PLOTS_DIR,
    IMAGE_CACHE_DIR, TARGET_EMOTION, USE_BINARY, NUM_CLASSES, CLASS_NAMES,
    BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, EARLY_STOPPING_PATIENCE,
    WEIGHT_DECAY, DROPOUT_RATE, IMAGE_SIZE, RANDOM_SEED,
    TRAIN_RATIO, VAL_RATIO,
)
from preprocessing.svc_parser import load_all_svc_files
from data.label_loader import load_labels
from data.dataset import get_dataloaders
from models.cnn_model import get_model
from evaluation.evaluate import evaluate_model, plot_confusion_matrix, plot_training_history


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Run one training epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Run validation."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def train(model_type: str = "cnn"):
    """
    Full training pipeline.

    Args:
        model_type: 'cnn' for EmotionCNN, 'resnet' for EmotionResNet.
    """
    print("=" * 60)
    print("  HANDWRITING EMOTION DETECTION — CNN TRAINING")
    print("=" * 60)

    # Set random seeds
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ─── Load Data ────────────────────────────────────────────────
    print(f"\nLoading dataset from: {DATASET_ROOT}")
    samples = load_all_svc_files(DATASET_ROOT)

    print(f"\nLoading labels (target: {TARGET_EMOTION}, binary: {USE_BINARY})")
    labels = load_labels(DASS_SCORES_PATH, TARGET_EMOTION, USE_BINARY)

    # Show class distribution
    print(f"Class distribution:")
    for idx, name in enumerate(CLASS_NAMES):
        count = sum(1 for v in labels.values() if v == idx)
        print(f"  {name}: {count} users")

    # ─── Create DataLoaders ───────────────────────────────────────
    train_loader, val_loader, test_loader = get_dataloaders(
        samples, labels,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        cache_dir=IMAGE_CACHE_DIR,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        random_seed=RANDOM_SEED,
    )

    # ─── Create Model ────────────────────────────────────────────
    model = get_model(model_type, num_classes=NUM_CLASSES, dropout=DROPOUT_RATE)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {model_type.upper()}")
    print(f"  Total params:     {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    # ─── Loss, Optimizer, Scheduler ───────────────────────────────
    # Compute class weights for imbalanced data
    label_values = list(labels.values())
    class_counts = np.bincount(label_values, minlength=NUM_CLASSES).astype(np.float32)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * NUM_CLASSES
    class_weights = torch.FloatTensor(class_weights).to(device)
    print(f"  Class weights: {class_weights.cpu().numpy()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # ─── Training Loop ────────────────────────────────────────────
    print(f"\nStarting training for {NUM_EPOCHS} epochs...")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Early stopping patience: {EARLY_STOPPING_PATIENCE}")
    print()

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Logging
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:3d}/{NUM_EPOCHS} │ "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} │ "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f} │ "
              f"LR: {lr:.2e}")

        # Checkpointing (save best model)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            # include the emotion in the filename so we can keep multiple models
            checkpoint_path = os.path.join(MODEL_DIR, f"best_{model_type}_{TARGET_EMOTION}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "model_type": model_type,
                "num_classes": NUM_CLASSES,
                "target_emotion": TARGET_EMOTION,
                "class_names": CLASS_NAMES,
            }, checkpoint_path)
            print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\n⚡ Early stopping at epoch {epoch} (no improvement for {EARLY_STOPPING_PATIENCE} epochs)")
                break

    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed:.1f}s")
    print(f"Best epoch: {best_epoch} (val_loss: {best_val_loss:.4f})")

    # ─── Plot Training History ────────────────────────────────────
    history_path = os.path.join(PLOTS_DIR, f"training_history_{model_type}.png")
    plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path=history_path)

    # ─── Evaluate on Test Set ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("  EVALUATING ON TEST SET")
    print("=" * 60)

    # Load best model (same naming convention)
    checkpoint = torch.load(
        os.path.join(MODEL_DIR, f"best_{model_type}_{TARGET_EMOTION}.pth"),
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    results = evaluate_model(model, test_loader, device, class_names=CLASS_NAMES)

    # Plot confusion matrix
    cm_path = os.path.join(PLOTS_DIR, f"confusion_matrix_{model_type}.png")
    plot_confusion_matrix(
        results["labels"], results["predictions"],
        CLASS_NAMES, save_path=cm_path
    )

    # Save final model for inference
    # save final model with emotion in filename as well
    final_path = os.path.join(MODEL_DIR, f"final_{model_type}_{TARGET_EMOTION}.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_type": model_type,
        "num_classes": NUM_CLASSES,
        "target_emotion": TARGET_EMOTION,
        "class_names": CLASS_NAMES,
        "test_accuracy": results["accuracy"],
        "test_f1": results["f1_score"],
    }, final_path)
    print(f"\nFinal model saved to: {final_path}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train CNN for Handwriting Emotion Detection")
    parser.add_argument("--model", type=str, default="cnn", choices=["cnn", "resnet"],
                        help="Model architecture: 'cnn' or 'resnet'")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs from config")
    args = parser.parse_args()

    if args.epochs:
        import config
        config.NUM_EPOCHS = args.epochs

    train(model_type=args.model)
