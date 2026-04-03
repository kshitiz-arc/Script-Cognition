"""
SVC File Parser for EMOTHAW Dataset.

Parses online handwriting recordings stored in .svc format.
Each file contains pen trajectory data: X, Y, timestamp, pen_status, azimuth, altitude, pressure.
"""
import os
import glob
import numpy as np


def parse_svc(filepath: str) -> np.ndarray:
    """
    Parse a single .svc file into a NumPy array.

    The first line contains the total number of data points.
    Subsequent lines contain 7 space-separated values:
        X, Y, timestamp, pen_status, azimuth, altitude, pressure

    Args:
        filepath: Absolute path to the .svc file.

    Returns:
        np.ndarray of shape (N, 7) with float64 values.
        Columns: [x, y, timestamp, pen_status, azimuth, altitude, pressure]
    """
    with open(filepath, "r") as f:
        lines = f.readlines()

    # First line is the number of data points
    num_points = int(lines[0].strip())

    data = []
    for line in lines[1: num_points + 1]:
        values = line.strip().split()
        if len(values) == 7:
            data.append([float(v) for v in values])

    return np.array(data, dtype=np.float64)


def load_all_svc_files(dataset_root: str) -> list[dict]:
    """
    Scan the EMOTHAW dataset and load all .svc file paths, organized by user.

    Args:
        dataset_root: Path to the DataEmothaw directory.

    Returns:
        List of dicts, each containing:
            - 'user_id': int (e.g., 1 for user00001)
            - 'collection': str ('Collection1' or 'Collection2')
            - 'task_id': int (1-7, the handwriting task number)
            - 'filepath': str (absolute path to .svc file)
            - 'data': np.ndarray (parsed pen data)
    """
    samples = []

    for collection in ["Collection1", "Collection2"]:
        collection_dir = os.path.join(dataset_root, collection)
        if not os.path.isdir(collection_dir):
            continue

        # Find all .svc files
        svc_pattern = os.path.join(collection_dir, "user*", "session*", "*.svc")
        svc_files = sorted(glob.glob(svc_pattern))

        for svc_path in svc_files:
            filename = os.path.basename(svc_path)

            # Skip backup/old files
            if "_old" in filename:
                continue

            # Extract user ID from path: .../user00001/session00001/...
            parts = svc_path.replace("\\", "/").split("/")
            user_dir = [p for p in parts if p.startswith("user")][0]
            user_id = int(user_dir.replace("user", ""))

            # Extract task ID from filename: u00001s00001_hw00003.svc → 3
            hw_part = filename.split("_hw")[1] if "_hw" in filename else "00001.svc"
            # Handle files with " - Copie" or similar suffixes
            hw_part = hw_part.split()[0] if " " in hw_part else hw_part
            task_id = int(hw_part.replace(".svc", ""))

            try:
                data = parse_svc(svc_path)
                if data.shape[0] > 0:
                    samples.append({
                        "user_id": user_id,
                        "collection": collection,
                        "task_id": task_id,
                        "filepath": svc_path,
                        "data": data,
                    })
            except Exception as e:
                print(f"Warning: Could not parse {svc_path}: {e}")

    print(f"Loaded {len(samples)} samples from {len(set(s['user_id'] for s in samples))} users")
    return samples


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import DATASET_ROOT

    samples = load_all_svc_files(DATASET_ROOT)
    if samples:
        s = samples[0]
        print(f"\nSample: user={s['user_id']}, task={s['task_id']}, collection={s['collection']}")
        print(f"  Data shape: {s['data'].shape}")
        print(f"  X range: [{s['data'][:, 0].min():.0f}, {s['data'][:, 0].max():.0f}]")
        print(f"  Y range: [{s['data'][:, 1].min():.0f}, {s['data'][:, 1].max():.0f}]")
        print(f"  Pen down points: {(s['data'][:, 3] == 1).sum()}")
        print(f"  Pen up points: {(s['data'][:, 3] == 0).sum()}")
