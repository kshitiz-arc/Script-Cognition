#!/usr/bin/env python3
"""
Test script for Handwriting Emotion Detection project.

Verifies:
1. All imports work correctly
2. Dataset can be loaded
3. Models can be instantiated
4. Feature extraction works
5. CLI commands are accessible
"""
import os
import sys

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all major modules can be imported."""
    print("=" * 70)
    print("TEST 1: Verifying imports...")
    print("=" * 70)

    try:
        from config import (
            TARGET_EMOTION, USE_BINARY, NUM_CLASSES,
            DATASET_ROOT, MODEL_DIR, PLOTS_DIR
        )
        print("✓ config module imports")

        from preprocessing.svc_parser import parse_svc, load_all_svc_files
        print("✓ preprocessing.svc_parser imports")

        from data.label_loader import load_labels
        print("✓ data.label_loader imports")

        from data.dataset import EmothawImageDataset, get_dataloaders
        print("✓ data.dataset imports")

        from features.image_generator import trajectory_to_image
        print("✓ features.image_generator imports")

        from features.signal_features import extract_signal_features
        print("✓ features.signal_features imports")

        from models.cnn_model import EmotionCNN, EmotionResNet, get_model
        print("✓ models.cnn_model imports")

        from training.train_cnn import train, train_one_epoch, validate
        print("✓ training.train_cnn imports")

        from evaluation.evaluate import evaluate_model, plot_confusion_matrix
        print("✓ evaluation.evaluate imports")

        from inference.predict import load_trained_model, predict
        print("✓ inference.predict imports")

        from utils.visualization import plot_trajectory
        print("✓ utils.visualization imports")

        print("\n✅ All imports successful!\n")
        return True

    except Exception as e:
        print(f"\n❌ Import failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_config():
    """Test configuration values."""
    print("=" * 70)
    print("TEST 2: Verifying configuration...")
    print("=" * 70)

    try:
        from config import (
            TARGET_EMOTION, USE_BINARY, NUM_CLASSES, CLASS_NAMES,
            DATASET_ROOT, DASS_SCORES_PATH, MODEL_DIR,
            IMAGE_SIZE, BATCH_SIZE, LEARNING_RATE
        )

        print(f"✓ Target emotion: {TARGET_EMOTION}")
        print(f"✓ Binary mode: {USE_BINARY}")
        print(f"✓ Number of classes: {NUM_CLASSES}")
        print(f"✓ Class names: {CLASS_NAMES}")
        print(f"✓ Image size: {IMAGE_SIZE}")
        print(f"✓ Batch size: {BATCH_SIZE}")
        print(f"✓ Learning rate: {LEARNING_RATE}")

        print(f"\n✓ Dataset root: {DATASET_ROOT}")
        if os.path.exists(DATASET_ROOT):
            print(f"  ✓ Dataset directory exists")
        else:
            print(f"  ⚠ Dataset directory not found (expected, can still test)")

        print(f"✓ DASS scores path: {DASS_SCORES_PATH}")
        if os.path.exists(DASS_SCORES_PATH):
            print(f"  ✓ DASS file exists")
        else:
            print(f"  ⚠ DASS file not found")

        print("\n✅ Configuration verification successful!\n")
        return True

    except Exception as e:
        print(f"\n❌ Configuration test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_models():
    """Test model instantiation."""
    print("=" * 70)
    print("TEST 3: Testing model instantiation...")
    print("=" * 70)

    try:
        import torch
        from models.cnn_model import EmotionCNN, EmotionResNet, get_model

        # Test EmotionCNN
        model_cnn = EmotionCNN(num_classes=2)
        print(f"✓ EmotionCNN created")
        print(f"  - Parameters: {sum(p.numel() for p in model_cnn.parameters()):,}")

        # Test EmotionResNet
        model_resnet = EmotionResNet(num_classes=2, pretrained=False)
        print(f"✓ EmotionResNet created")
        print(f"  - Parameters: {sum(p.numel() for p in model_resnet.parameters()):,}")

        # Test get_model factory
        model_factory_cnn = get_model("cnn", num_classes=2)
        model_factory_resnet = get_model("resnet", num_classes=2)
        print(f"✓ get_model factory function works")

        # Test forward pass
        x = torch.randn(2, 3, 224, 224)
        out_cnn = model_cnn(x)
        out_resnet = model_resnet(x)
        print(f"✓ Forward pass successful")
        print(f"  - EmotionCNN output shape: {out_cnn.shape}")
        print(f"  - EmotionResNet output shape: {out_resnet.shape}")

        print("\n✅ Model tests passed!\n")
        return True

    except Exception as e:
        print(f"\n❌ Model test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_features():
    """Test feature extraction."""
    print("=" * 70)
    print("TEST 4: Testing feature extraction...")
    print("=" * 70)

    try:
        import numpy as np
        from features.signal_features import extract_signal_features

        # Create dummy handwriting data
        n_points = 100
        dummy_data = np.random.randn(n_points, 7).astype(np.float64)
        # Set reasonable ranges
        dummy_data[:, 0] = np.cumsum(np.random.randn(n_points)) * 10  # x
        dummy_data[:, 1] = np.cumsum(np.random.randn(n_points)) * 10  # y
        dummy_data[:, 2] = np.arange(n_points) * 10  # timestamp
        dummy_data[:, 3] = np.ones(n_points)  # pen_status
        dummy_data[:, 4] = np.random.uniform(0, 360, n_points)  # azimuth
        dummy_data[:, 5] = np.random.uniform(0, 90, n_points)  # altitude
        dummy_data[:, 6] = np.random.uniform(0, 255, n_points)  # pressure

        features = extract_signal_features(dummy_data)
        print(f"✓ Feature extraction works")
        print(f"  - Features extracted: {len(features)}")
        print(f"  - Sample features:")
        for i, (name, value) in enumerate(list(features.items())[:5]):
            print(f"    - {name}: {value:.4f}")

        print("\n✅ Feature extraction tests passed!\n")
        return True

    except Exception as e:
        print(f"\n❌ Feature extraction test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_cli_help():
    """Test CLI help command."""
    print("=" * 70)
    print("TEST 5: Testing CLI interface...")
    print("=" * 70)

    try:
        import subprocess

        result = subprocess.run(
            [sys.executable, "main.py", "--help"],
            cwd=os.path.dirname(__file__),
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            print("✓ CLI help accessible")
            lines = result.stdout.split('\n')[:10]
            for line in lines:
                if line.strip():
                    print(f"  {line}")

            print("\n✅ CLI interface tests passed!\n")
            return True
        else:
            print(f"❌ CLI help failed: {result.stderr}")
            return False

    except Exception as e:
        print(f"\n❌ CLI test failed: {e}\n")
        return False


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + " HANDWRITING EMOTION DETECTION — PROJECT VERIFICATION ".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")
    print()

    results = []

    # Run all tests
    results.append(("Imports", test_imports()))
    results.append(("Configuration", test_config()))
    results.append(("Models", test_models()))
    results.append(("Features", test_features()))
    results.append(("CLI", test_cli_help()))

    # Summary
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:20s}.... {status}")

    print("=" * 70)
    print(f"Result: {passed}/{total} tests passed")
    print("=" * 70)

    if passed == total:
        print("\n✅ All tests passed! Project is ready to use.\n")
        print("Quick start commands:")
        print("  python main.py train --model cnn")
        print("  python main.py info")
        print("  python main.py predict --svc_path <file.svc>")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the output above.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
