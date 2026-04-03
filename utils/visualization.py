"""
Visualization utilities for handwriting emotion detection.

Provides plotting functions for pen trajectories, training metrics,
and model performance analysis.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_trajectory(data: np.ndarray, title: str = "Handwriting Trajectory",
                    save_path: str = None, show_pressure: bool = True):
    """
    Visualize a pen trajectory with color-coded pen status and pressure.

    Args:
        data: np.ndarray of shape (N, 7) — [x, y, timestamp, pen_status, azimuth, altitude, pressure]
        title: Plot title.
        save_path: If provided, save plot to this path.
        show_pressure: If True, color-code by pressure; otherwise by pen status.
    """
    fig, axes = plt.subplots(1, 2 if show_pressure else 1,
                              figsize=(14 if show_pressure else 7, 6))

    if not show_pressure:
        axes = [axes]

    x = data[:, 0]
    y = data[:, 1]
    pen_status = data[:, 3]
    pressure = data[:, 6]

    # Plot 1: Pen status (blue = pen down, red = pen up)
    ax1 = axes[0]
    pen_down = pen_status == 1
    pen_up = pen_status == 0

    ax1.scatter(x[pen_down], -y[pen_down], c="navy", s=1, alpha=0.8, label="Pen Down")
    ax1.scatter(x[pen_up], -y[pen_up], c="red", s=0.5, alpha=0.3, label="Pen Up")
    ax1.set_title(f"{title}\n(Pen Status)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.legend(markerscale=5)
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.2)

    if show_pressure:
        # Plot 2: Pressure heatmap
        ax2 = axes[1]
        valid = pen_down & (pressure > 0)
        scatter = ax2.scatter(x[valid], -y[valid], c=pressure[valid],
                              cmap="viridis", s=2, alpha=0.8)
        plt.colorbar(scatter, ax=ax2, label="Pressure")
        ax2.set_title(f"{title}\n(Pressure)", fontsize=12, fontweight="bold")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_aspect("equal")
        ax2.grid(True, alpha=0.2)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Trajectory plot saved to: {save_path}")

    plt.close()


def plot_dataset_statistics(labels: dict, class_names: list,
                            target_emotion: str, save_path: str = None):
    """
    Plot class distribution of the dataset.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    counts = {}
    for label in labels.values():
        cls_name = class_names[label]
        counts[cls_name] = counts.get(cls_name, 0) + 1

    names = list(counts.keys())
    values = list(counts.values())
    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))

    bars = ax.bar(names, values, color=colors, edgecolor="black", linewidth=0.5)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(val), ha="center", va="bottom", fontweight="bold")

    ax.set_title(f"Class Distribution — {target_emotion.capitalize()}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of Users")
    ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Dataset statistics saved to: {save_path}")

    plt.close()


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import DATASET_ROOT, PLOTS_DIR, TARGET_EMOTION, USE_BINARY, CLASS_NAMES
    from preprocessing.svc_parser import parse_svc, load_all_svc_files
    from data.label_loader import load_labels

    # Plot a sample trajectory
    test_file = os.path.join(
        DATASET_ROOT, "Collection1", "user00001", "session00001", "u00001s00001_hw00001.svc"
    )
    data = parse_svc(test_file)
    plot_trajectory(data, title="User 1 — Task 1",
                    save_path=os.path.join(PLOTS_DIR, "sample_trajectory.png"))

    # Plot dataset statistics
    labels = load_labels(target_emotion=TARGET_EMOTION, use_binary=USE_BINARY)
    plot_dataset_statistics(labels, CLASS_NAMES, TARGET_EMOTION,
                           save_path=os.path.join(PLOTS_DIR, "class_distribution.png"))
