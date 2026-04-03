"""
Neural network models for emotion classification from handwriting.
"""

from .cnn_model import EmotionCNN, EmotionResNet, get_model

__all__ = ["EmotionCNN", "EmotionResNet", "get_model"]
