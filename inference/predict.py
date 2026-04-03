"""
Inference module for Handwriting Emotion Detection.

Takes a .svc file path and outputs the predicted emotional state
with confidence score.
"""
import os
import sys
import argparse
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.svc_parser import parse_svc
from features.image_generator import trajectory_to_image
from models.cnn_model import get_model
from config import MODEL_DIR, IMAGE_SIZE, NUM_CLASSES, CLASS_NAMES, TARGET_EMOTION


def load_trained_model(model_path: str = None, device=None, target_emotion: str = None):
    """
    Load a trained model from checkpoint.

    Args:
        model_path: Path to .pth checkpoint. If None, uses default best model for the
            requested emotion (or config.TARGET_EMOTION if not provided).
        device: torch device. If None, auto-detects.
        target_emotion: Override which emotion-specific model to load. If None,
            uses the value from config.TARGET_EMOTION.

    Returns:
        (model, checkpoint_info)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if target_emotion is None:
        target_emotion = TARGET_EMOTION

    # If a specific path isn't given, search for emotion-tagged files
    if model_path is None:
        patterns = [
            f"best_cnn_{target_emotion}.pth",
            f"final_cnn_{target_emotion}.pth",
            f"best_resnet_{target_emotion}.pth",
            f"final_resnet_{target_emotion}.pth",
        ]
        for name in patterns:
            candidate = os.path.join(MODEL_DIR, name)
            if os.path.exists(candidate):
                model_path = candidate
                break
        # if we didn't find an emotion-specific model, fall back to legacy names
        if model_path is None:
            legacy = [
                "best_cnn_model.pth",
                "final_cnn_model.pth",
                "best_resnet_model.pth",
                "final_resnet_model.pth",
            ]
            for name in legacy:
                candidate = os.path.join(MODEL_DIR, name)
                if os.path.exists(candidate):
                    model_path = candidate
                    break

    if model_path is None or not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model found for emotion '{target_emotion}'. "
            "Please train a model first.\n"
            f"Searched in: {MODEL_DIR}"
        )

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model_type = checkpoint.get("model_type", "cnn")
    num_classes = checkpoint.get("num_classes", NUM_CLASSES)
    class_names = checkpoint.get("class_names", CLASS_NAMES)
    target_emotion = checkpoint.get("target_emotion", TARGET_EMOTION)

    model = get_model(model_type, num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    info = {
        "model_type": model_type,
        "num_classes": num_classes,
        "class_names": class_names,
        "target_emotion": target_emotion,
        "model_path": model_path,
    }

    return model, info


def predict(svc_path: str, model=None, model_info=None, model_path: str = None):
    """
    Predict the emotional state from a handwriting .svc file.

    Args:
        svc_path: Path to the .svc file.
        model: Pre-loaded model. If None, loads from model_path.
        model_info: Info dict from load_trained_model.
        model_path: Path to model checkpoint.

    Returns:
        Dict with prediction results.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model is None:
        model, model_info = load_trained_model(model_path, device)

    # Parse SVC file
    data = parse_svc(svc_path)
    if data.shape[0] == 0:
        return {"error": "Empty or invalid .svc file"}

    # Generate trajectory image
    img = trajectory_to_image(data, image_size=IMAGE_SIZE)

    # Convert to tensor
    img_tensor = torch.from_numpy(img).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).repeat(3, 1, 1)  # (3, H, W)
    img_tensor = img_tensor.unsqueeze(0)  # (1, 3, H, W)
    img_tensor = img_tensor.to(device)

    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_idx].item()

    class_names = model_info.get("class_names", CLASS_NAMES)
    target_emotion = model_info.get("target_emotion", TARGET_EMOTION)

    result = {
        "predicted_class": class_names[pred_idx],
        "predicted_index": pred_idx,
        "confidence": confidence,
        "target_emotion": target_emotion,
        "all_probabilities": {
            name: probs[0, i].item()
            for i, name in enumerate(class_names)
        },
        "svc_path": svc_path,
        "num_data_points": data.shape[0],
    }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Predict emotional state from handwriting .svc file"
    )
    parser.add_argument("--svc_path", type=str, required=True,
                        help="Path to the .svc handwriting file")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to trained model checkpoint (.pth)")
    parser.add_argument("--emotion", type=str, default=None,
                        help="Target emotion (overrides model's stored setting)")
    args = parser.parse_args()

    if not os.path.exists(args.svc_path):
        print(f"Error: File not found: {args.svc_path}")
        sys.exit(1)

    print("=" * 50)
    print("  HANDWRITING EMOTION PREDICTION")
    print("=" * 50)
    print(f"\nInput file: {args.svc_path}")

    try:
        result = predict(args.svc_path, model_path=args.model_path)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        sys.exit(1)

    if "error" in result:
        print(f"\nError: {result['error']}")
        sys.exit(1)

    print(f"Data points: {result['num_data_points']}")
    print(f"Target dimension: {result['target_emotion'].capitalize()}")
    print()
    print("─" * 40)
    print(f"  Predicted Emotion: {result['predicted_class']}")
    print(f"  Confidence: {result['confidence']:.2f}")
    print("─" * 40)
    print()
    print("Class Probabilities:")
    for cls_name, prob in result["all_probabilities"].items():
        bar = "█" * int(prob * 30)
        print(f"  {cls_name:20s}: {prob:.4f}  {bar}")


if __name__ == "__main__":
    main()
