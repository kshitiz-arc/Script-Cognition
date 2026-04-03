"""
CNN Model for Handwriting Emotion Classification.

Provides two architectures:
1. EmotionCNN — A lightweight custom CNN for small datasets.
2. EmotionResNet — Transfer learning with ResNet18.
"""
import torch
import torch.nn as nn
import torchvision.models as models


class EmotionCNN(nn.Module):
    """
    Lightweight CNN with 4 convolutional blocks + fully connected classifier.
    Designed for 224x224 grayscale (3-channel replicated) handwriting images.
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.5):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 3 → 32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 224 → 112

            # Block 2: 32 → 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 112 → 56

            # Block 3: 64 → 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 56 → 28

            # Block 4: 128 → 256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),  # → 4x4
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class EmotionResNet(nn.Module):
    """
    ResNet18-based model with transfer learning.
    Replaces the final FC layer for emotion classification.
    Optionally freezes early layers for fine-tuning.
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True,
                 freeze_backbone: bool = True):
        super().__init__()

        # Load pretrained ResNet18
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)

        # Freeze backbone layers if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            # Unfreeze layer4 and fc for fine-tuning
            for param in self.backbone.layer4.parameters():
                param.requires_grad = True

        # Replace final FC layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


def get_model(model_type: str = "cnn", num_classes: int = 2, **kwargs) -> nn.Module:
    """
    Factory function to create a model.

    Args:
        model_type: 'cnn' for EmotionCNN, 'resnet' for EmotionResNet.
        num_classes: Number of output classes.

    Returns:
        nn.Module instance.
    """
    if model_type == "cnn":
        return EmotionCNN(num_classes=num_classes, **kwargs)
    elif model_type == "resnet":
        return EmotionResNet(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'cnn' or 'resnet'.")


if __name__ == "__main__":
    # Quick test
    model = EmotionCNN(num_classes=2)
    x = torch.randn(4, 3, 224, 224)
    out = model(x)
    print(f"EmotionCNN output shape: {out.shape}")  # (4, 2)

    model2 = EmotionResNet(num_classes=2, pretrained=False)
    out2 = model2(x)
    print(f"EmotionResNet output shape: {out2.shape}")  # (4, 2)

    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nEmotionCNN: {total:,} total params, {trainable:,} trainable")

    total2 = sum(p.numel() for p in model2.parameters())
    trainable2 = sum(p.numel() for p in model2.parameters() if p.requires_grad)
    print(f"EmotionResNet: {total2:,} total params, {trainable2:,} trainable")
