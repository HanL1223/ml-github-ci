import sys
import os
import pickle

# Add repo root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from train import train_model

def test_train_model_creates_file():
    if os.path.exists("model.pkl"):
        os.remove("model.pkl")
    train_model()
    assert os.path.exists("model.pkl")