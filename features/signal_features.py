"""
Signal-Based Feature Extraction for Handwriting.

Extracts numerical features directly from pen trajectory data:
- Writing speed (velocity statistics)
- Pen pressure patterns
- Stroke curvature
- Pen lifts and pauses
- Acceleration characteristics
"""
import numpy as np
from scipy import signal as scipy_signal


def compute_velocity(data: np.ndarray) -> np.ndarray:
    """
    Compute instantaneous velocity from x, y coordinates.

    Args:
        data: np.ndarray of shape (N, 7) — [x, y, timestamp, pen_status, azimuth, altitude, pressure]

    Returns:
        np.ndarray of velocities, shape (N,)
    """
    x = data[:, 0]
    y = data[:, 1]
    t = data[:, 2]

    # Compute distances
    dx = np.diff(x)
    dy = np.diff(y)
    distances = np.sqrt(dx ** 2 + dy ** 2)

    # Compute time differences
    dt = np.diff(t)
    dt = np.where(dt == 0, 1e-6, dt)  # Avoid division by zero

    # Compute velocities
    velocities = distances / dt

    # Pad first element
    velocities = np.concatenate([[velocities[0]], velocities])

    return velocities


def compute_acceleration(data: np.ndarray) -> np.ndarray:
    """
    Compute acceleration from velocity.

    Args:
        data: np.ndarray of shape (N, 7)

    Returns:
        np.ndarray of accelerations, shape (N,)
    """
    velocities = compute_velocity(data)
    t = data[:, 2]
    dt = np.diff(t)
    dt = np.where(dt == 0, 1e-6, dt)

    accelerations = np.diff(velocities) / dt

    # Pad
    accelerations = np.concatenate([[accelerations[0]], accelerations])

    return accelerations


def compute_curvature(data: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Compute trajectory curvature using angle changes.

    Args:
        data: np.ndarray of shape (N, 7)
        window: Window size for angle computation

    Returns:
        np.ndarray of curvatures, shape (N,)
    """
    x = data[:, 0]
    y = data[:, 1]
    
    n = len(x)
    
    if n < 2:
        return np.zeros(n)

    # Compute angles
    dx = np.diff(x)
    dy = np.diff(y)
    angles = np.arctan2(dy, dx)

    # Compute angle changes
    if len(angles) < 2:
        curvature = np.zeros(n)
    else:
        angle_diffs = np.abs(np.diff(angles))
        # Wrap to [0, pi]
        angle_diffs = np.minimum(angle_diffs, 2 * np.pi - angle_diffs)

        # Smooth with moving average
        if len(angle_diffs) > 0:
            curvature_smooth = np.convolve(angle_diffs, np.ones(window) / window, mode="same")
            # Pad to match original length: we lost 2 points via 2x diff
            curvature = np.concatenate([[curvature_smooth[0]], curvature_smooth, [curvature_smooth[-1]]])[:n]
        else:
            curvature = np.zeros(n)

    return curvature


def count_pen_lifts(data: np.ndarray) -> int:
    """
    Count the number of pen lift events.

    Args:
        data: np.ndarray of shape (N, 7)

    Returns:
        int — number of pen lifts
    """
    pen_status = data[:, 3]
    # Pen lift is transition from 1 to 0
    transitions = np.diff(pen_status.astype(int))
    lifts = np.sum(transitions == -1)
    return int(lifts)


def extract_signal_features(data: np.ndarray) -> dict:
    """
    Extract comprehensive handwriting signal features.

    Args:
        data: np.ndarray of shape (N, 7) — [x, y, timestamp, pen_status, azimuth, altitude, pressure]

    Returns:
        dict with feature names → values
    """
    x = data[:, 0]
    y = data[:, 1]
    t = data[:, 2]
    pen_status = data[:, 3]
    pressure = data[:, 6]

    # Filter to pen-down segments
    pen_down_idx = pen_status == 1
    x_down = x[pen_down_idx]
    y_down = y[pen_down_idx]
    pressure_down = pressure[pen_down_idx]
    t_down = t[pen_down_idx]

    features = {}

    # ─── Basic Statistics ───────────────────────────────────────────────
    features["total_duration"] = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    features["num_points"] = int(len(data))
    features["num_pen_downs"] = int(np.sum(pen_status == 1))

    # ─── Velocity Features ──────────────────────────────────────────────
    velocities = compute_velocity(data)
    vel_down = velocities[pen_down_idx] if len(velocities[pen_down_idx]) > 0 else np.array([0.0])

    features["mean_velocity"] = float(np.mean(vel_down)) if len(vel_down) > 0 else 0.0
    features["max_velocity"] = float(np.max(vel_down)) if len(vel_down) > 0 else 0.0
    features["std_velocity"] = float(np.std(vel_down)) if len(vel_down) > 0 else 0.0
    features["median_velocity"] = float(np.median(vel_down)) if len(vel_down) > 0 else 0.0

    # ─── Acceleration Features ─────────────────────────────────────────
    accelerations = compute_acceleration(data)
    acc_down = np.abs(accelerations[pen_down_idx]) if len(accelerations[pen_down_idx]) > 0 else np.array([0.0])

    features["mean_acceleration"] = float(np.mean(acc_down)) if len(acc_down) > 0 else 0.0
    features["max_acceleration"] = float(np.max(acc_down)) if len(acc_down) > 0 else 0.0
    features["std_acceleration"] = float(np.std(acc_down)) if len(acc_down) > 0 else 0.0

    # ─── Pressure Features ──────────────────────────────────────────────
    features["mean_pressure"] = float(np.mean(pressure_down)) if len(pressure_down) > 0 else 0.0
    features["max_pressure"] = float(np.max(pressure_down)) if len(pressure_down) > 0 else 0.0
    features["std_pressure"] = float(np.std(pressure_down)) if len(pressure_down) > 0 else 0.0
    features["pressure_variance"] = float(np.var(pressure_down)) if len(pressure_down) > 0 else 0.0

    # ─── Curvature Features ─────────────────────────────────────────────
    curvature = compute_curvature(data)
    curv_down = curvature[pen_down_idx] if len(curvature[pen_down_idx]) > 0 else np.array([0.0])

    features["mean_curvature"] = float(np.mean(curv_down)) if len(curv_down) > 0 else 0.0
    features["max_curvature"] = float(np.max(curv_down)) if len(curv_down) > 0 else 0.0
    features["std_curvature"] = float(np.std(curv_down)) if len(curv_down) > 0 else 0.0

    # ─── Stroke Length & Coverage ──────────────────────────────────────
    if len(x_down) > 1:
        stroke_distances = np.sqrt(np.diff(x_down) ** 2 + np.diff(y_down) ** 2)
        features["total_stroke_length"] = float(np.sum(stroke_distances))
        features["mean_stroke_segment"] = float(np.mean(stroke_distances))
    else:
        features["total_stroke_length"] = 0.0
        features["mean_stroke_segment"] = 0.0

    # Spatial coverage (bounding box)
    x_range = float(np.ptp(x_down)) if len(x_down) > 0 else 0.0
    y_range = float(np.ptp(y_down)) if len(y_down) > 0 else 0.0
    features["bounding_width"] = x_range
    features["bounding_height"] = y_range
    features["bounding_area"] = x_range * y_range
    features["aspect_ratio"] = (x_range / y_range) if y_range > 0 else 0.0

    # ─── Pen Lifts ─────────────────────────────────────────────────────
    features["num_pen_lifts"] = int(count_pen_lifts(data))

    # Duration of pen lifts (pauses)
    pen_up_idx = pen_status == 0
    if np.any(pen_up_idx):
        t_up = t[pen_up_idx]
        # Compute pause durations (time between consecutive pen downs)
        pen_up_durations = np.diff(t_up)
        features["mean_pause_duration"] = float(np.mean(pen_up_durations)) if len(pen_up_durations) > 0 else 0.0
        features["max_pause_duration"] = float(np.max(pen_up_durations)) if len(pen_up_durations) > 0 else 0.0
        features["total_pause_time"] = float(np.sum(pen_up_durations))
    else:
        features["mean_pause_duration"] = 0.0
        features["max_pause_duration"] = 0.0
        features["total_pause_time"] = 0.0

    # ─── Efficiency & Consistency ──────────────────────────────────────
    if features["total_duration"] > 0:
        features["writing_ratio"] = features["num_pen_downs"] / features["total_duration"]
    else:
        features["writing_ratio"] = 0.0

    # Entropy of pressure (variability)
    if len(pressure_down) > 1:
        hist, _ = np.histogram(pressure_down, bins=10)
        hist = hist[hist > 0]
        p = hist / hist.sum()
        features["pressure_entropy"] = float(-np.sum(p * np.log2(p + 1e-10)))
    else:
        features["pressure_entropy"] = 0.0

    return features


def extract_batch_features(samples: list) -> np.ndarray:
    """
    Extract signal features for a batch of samples.

    Args:
        samples: List of sample dicts with 'data' key containing pen trajectory arrays.

    Returns:
        np.ndarray of shape (n_samples, n_features)
    """
    all_features = []
    feature_names = None

    for sample in samples:
        features_dict = extract_signal_features(sample["data"])
        if feature_names is None:
            feature_names = list(features_dict.keys())
        features_list = [features_dict[name] for name in feature_names]
        all_features.append(features_list)

    return np.array(all_features), feature_names


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import DATASET_ROOT
    from preprocessing.svc_parser import parse_svc

    # Test feature extraction
    test_file = os.path.join(
        DATASET_ROOT, "Collection1", "user00001", "session00001", "u00001s00001_hw00001.svc"
    )
    data = parse_svc(test_file)

    features = extract_signal_features(data)
    print("Extracted Signal Features:")
    print("=" * 50)
    for name, value in features.items():
        print(f"  {name:30s}: {value:12.4f}")

    print(f"\nTotal features extracted: {len(features)}")
