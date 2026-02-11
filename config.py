"""
Configuration for Brazilian Document Classification Model
"""
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
MODEL_SAVE_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"

# Dataset split: 60% training, 40% testing
TRAIN_RATIO = 0.60
TEST_RATIO = 0.40
RANDOM_SEED = 42

# Training hyperparameters
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-3
IMG_SIZE = (224, 224)

# Model
NUM_CLASSES = 9  # Adjust if your dataset has 8 or 9 types
MODEL_NAME = "resnet18"  # or "efficientnet_b0", "convnext_tiny"
