"""
Dataset handling with 60/40 train-test split for Brazilian Document Classification
"""
import os
import shutil
from pathlib import Path
from typing import Tuple

import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms


class BrazilianDocumentDataset(Dataset):
    """PyTorch Dataset for Brazilian document images."""

    def __init__(self, root_dir: Path, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        self._load_samples()

    def _load_samples(self):
        """Load image paths and labels from directory structure: root/class_name/*.jpg"""
        if not self.root_dir.exists():
            return

        classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

        for cls in classes:
            cls_dir = self.root_dir / cls
            for img_path in cls_dir.glob("*"):
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                    self.samples.append((str(img_path), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms(img_size: Tuple[int, int], is_train: bool = True):
    """Get image transforms for training or evaluation."""
    if is_train:
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])


def split_dataset(
    raw_data_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.60,
    random_seed: int = 42,
):
    """
    Split raw dataset 60/40 into train and test directories.
    Expected structure: raw_data_dir/class_name/*.jpg
    """
    raw_data_dir = Path(raw_data_dir)
    output_dir = Path(output_dir)
    train_dir = output_dir / "train"
    test_dir = output_dir / "test"

    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    for class_dir in raw_data_dir.iterdir():
        if not class_dir.is_dir():
            continue

        images = list(class_dir.glob("*"))
        images = [p for p in images if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]

        if not images:
            continue

        train_imgs, test_imgs = train_test_split(
            images, train_size=train_ratio, random_state=random_seed, stratify=None
        )

        (train_dir / class_dir.name).mkdir(exist_ok=True)
        (test_dir / class_dir.name).mkdir(exist_ok=True)

        for img in train_imgs:
            shutil.copy2(img, train_dir / class_dir.name / img.name)
        for img in test_imgs:
            shutil.copy2(img, test_dir / class_dir.name / img.name)

        print(f"  {class_dir.name}: {len(train_imgs)} train, {len(test_imgs)} test")
