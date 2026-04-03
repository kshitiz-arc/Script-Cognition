"""
Inference module for emotion prediction from handwriting.
"""

from .predict import load_trained_model, predict

__all__ = ["load_trained_model", "predict"]
