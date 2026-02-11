"""
Split raw Brazilian document dataset: 60% train, 40% test.
Place your raw data in: data/raw/<document_type>/<images>
"""
from pathlib import Path

from config import DATA_DIR, TRAIN_RATIO, RANDOM_SEED
from src.dataset import split_dataset

RAW_DATA_DIR = DATA_DIR / "raw"


def main():
    print("Brazilian Document Dataset Split (60% train / 40% test)")
    print("=" * 50)

    if not RAW_DATA_DIR.exists():
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        print(f"\nCreated: {RAW_DATA_DIR}")
        print("Please place your dataset with structure:")
        print("  data/raw/CNH/*.jpg")
        print("  data/raw/CPF/*.jpg")
        print("  data/raw/RG/*.jpg")
        print("  ... (one folder per document type)")
        return

    print(f"Source: {RAW_DATA_DIR}")
    print(f"Output: {DATA_DIR}/train and {DATA_DIR}/test")
    print("\nSplitting classes:")
    split_dataset(RAW_DATA_DIR, DATA_DIR, train_ratio=TRAIN_RATIO, random_seed=RANDOM_SEED)
    print("\nDone! Run train.py to train the model.")


if __name__ == "__main__":
    main()
