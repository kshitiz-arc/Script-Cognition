"""
Trajectory-to-Image Generator.

Converts pen trajectory data from .svc files into grayscale images
suitable for CNN input. Draws pen strokes with varying thickness
based on pressure data.
"""
import numpy as np
import cv2


def trajectory_to_image(data: np.ndarray, image_size: int = 224,
                         line_thickness: int = 2, padding: int = 10) -> np.ndarray:
    """
    Render pen trajectory as a grayscale image.

    Args:
        data: np.ndarray of shape (N, 7) — [x, y, timestamp, pen_status, azimuth, altitude, pressure]
        image_size: Output image dimension (square).
        line_thickness: Base line width for pen strokes.
        padding: Pixel padding around the drawing.

    Returns:
        np.ndarray of shape (image_size, image_size) with uint8 values [0-255].
        White strokes on black background.
    """
    # Create black canvas
    img = np.zeros((image_size, image_size), dtype=np.uint8)

    x = data[:, 0]
    y = data[:, 1]
    pen_status = data[:, 3]
    pressure = data[:, 6]

    # Normalize coordinates to fit within image (with padding)
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    x_range = x_max - x_min if x_max > x_min else 1.0
    y_range = y_max - y_min if y_max > y_min else 1.0

    # Maintain aspect ratio
    scale = (image_size - 2 * padding) / max(x_range, y_range)

    x_norm = ((x - x_min) * scale + padding).astype(np.int32)
    y_norm = ((y - y_min) * scale + padding).astype(np.int32)

    # Flip Y axis (screen coordinates: top-left origin)
    y_norm = image_size - 1 - y_norm

    # Clamp to image bounds
    x_norm = np.clip(x_norm, 0, image_size - 1)
    y_norm = np.clip(y_norm, 0, image_size - 1)

    # Normalize pressure for line thickness variation
    p_min, p_max = pressure.min(), pressure.max()
    if p_max > p_min:
        p_norm = (pressure - p_min) / (p_max - p_min)
    else:
        p_norm = np.ones_like(pressure) * 0.5

    # Draw strokes — only when pen is down (pen_status == 1)
    for i in range(1, len(data)):
        if pen_status[i] == 1 and pen_status[i - 1] == 1:
            pt1 = (int(x_norm[i - 1]), int(y_norm[i - 1]))
            pt2 = (int(x_norm[i]), int(y_norm[i]))

            # Thickness varies with pressure
            thickness = max(1, int(line_thickness * (0.5 + p_norm[i])))

            # Intensity varies with pressure (brighter = higher pressure)
            intensity = int(150 + 105 * p_norm[i])

            cv2.line(img, pt1, pt2, intensity, thickness, cv2.LINE_AA)

    return img


def generate_dataset_images(samples: list, image_size: int = 224,
                            cache_dir: str = None) -> list[np.ndarray]:
    """
    Generate images for all samples. Optionally caches to disk.

    Args:
        samples: List of sample dicts from svc_parser.load_all_svc_files().
        image_size: Output image size.
        cache_dir: If provided, save/load images as PNG files.

    Returns:
        List of grayscale images as numpy arrays.
    """
    import os
    images = []

    for i, sample in enumerate(samples):
        cache_path = None
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(
                cache_dir,
                f"u{sample['user_id']:05d}_t{sample['task_id']:05d}.png"
            )
            if os.path.exists(cache_path):
                img = cv2.imread(cache_path, cv2.IMREAD_GRAYSCALE)
                images.append(img)
                continue

        img = trajectory_to_image(sample["data"], image_size=image_size)
        images.append(img)

        if cache_path:
            cv2.imwrite(cache_path, img)

    return images


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import DATASET_ROOT, IMAGE_SIZE
    from preprocessing.svc_parser import parse_svc

    # Test with a single file
    test_file = os.path.join(
        DATASET_ROOT, "Collection1", "user00001", "session00001", "u00001s00001_hw00001.svc"
    )
    data = parse_svc(test_file)
    img = trajectory_to_image(data, image_size=IMAGE_SIZE)
    print(f"Generated image shape: {img.shape}, dtype: {img.dtype}")
    print(f"Pixel range: [{img.min()}, {img.max()}]")

    # Save test image
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "outputs", "plots", "sample_trajectory.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)
    print(f"Saved sample image to: {output_path}")
