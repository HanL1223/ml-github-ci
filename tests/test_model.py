# tests/test_model.py
import sys
import os

# Ensure repo root is in Python path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from train import train_model

def test_train_model_creates_file():
    """Test that train_model creates model.pkl"""
    model_path = os.path.join(os.getcwd(), "model.pkl")

    # Remove existing model if it exists
    if os.path.exists(model_path):
        os.remove(model_path)

    # Train model
    train_model()

    # Check if model file exists
    assert os.path.exists(model_path)
