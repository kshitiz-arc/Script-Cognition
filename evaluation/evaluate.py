"""
Evaluation module for emotion classification.

Computes accuracy, precision, recall, F1-score, and generates
confusion matrix plots.
"""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def evaluate_model(model, dataloader, device, class_names=None):
    """
    Evaluate a PyTorch model on a dataloader.

    Args:
        model: Trained PyTorch model.
        dataloader: Test/val DataLoader.
        device: torch device.
        class_names: List of class name strings.

    Returns:
        Dict with metrics: accuracy, precision, recall, f1, predictions, labels.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Compute metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    rec = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    results = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "predictions": all_preds,
        "labels": all_labels,
        "probabilities": all_probs,
    }

    # Print classification report
    if class_names is None:
        class_names = [str(i) for i in range(len(np.unique(all_labels)))]

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))
    print(f"Overall Accuracy: {acc:.4f}")
    print(f"Macro Precision:  {prec:.4f}")
    print(f"Macro Recall:     {rec:.4f}")
    print(f"Macro F1-Score:   {f1:.4f}")
    print("=" * 60)

    return results


def plot_confusion_matrix(labels, predictions, class_names, save_path=None):
    """
    Generate and optionally save a confusion matrix plot.
    """
    cm = confusion_matrix(labels, predictions)
    cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw counts
    ax1 = axes[0]
    im1 = ax1.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax1.set_title("Confusion Matrix (Counts)", fontsize=14, fontweight="bold")
    plt.colorbar(im1, ax=ax1)
    ax1.set_xticks(range(len(class_names)))
    ax1.set_yticks(range(len(class_names)))
    ax1.set_xticklabels(class_names, rotation=45, ha="right")
    ax1.set_yticklabels(class_names)
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("True")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax1.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=14)

    # Normalized
    ax2 = axes[1]
    im2 = ax2.imshow(cm_normalized, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=1)
    ax2.set_title("Confusion Matrix (Normalized)", fontsize=14, fontweight="bold")
    plt.colorbar(im2, ax=ax2)
    ax2.set_xticks(range(len(class_names)))
    ax2.set_yticks(range(len(class_names)))
    ax2.set_xticklabels(class_names, rotation=45, ha="right")
    ax2.set_yticklabels(class_names)
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("True")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val = cm_normalized[i, j]
            color = "white" if val > 0.5 else "black"
            ax2.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=14)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved to: {save_path}")

    plt.close()


def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path=None):
    """
    Plot training and validation loss/accuracy curves.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(train_losses) + 1)

    # Loss
    ax1.plot(epochs, train_losses, "b-o", label="Train Loss", markersize=3)
    ax1.plot(epochs, val_losses, "r-o", label="Val Loss", markersize=3)
    ax1.set_title("Training & Validation Loss", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, train_accs, "b-o", label="Train Accuracy", markersize=3)
    ax2.plot(epochs, val_accs, "r-o", label="Val Accuracy", markersize=3)
    ax2.set_title("Training & Validation Accuracy", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Training history saved to: {save_path}")

    plt.close()
