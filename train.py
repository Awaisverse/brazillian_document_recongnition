"""
Train Brazilian Document Classification Model (60% of dataset)
"""
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import (
    BATCH_SIZE,
    EPOCHS,
    IMG_SIZE,
    LEARNING_RATE,
    MODEL_SAVE_DIR,
    TRAIN_DIR,
)
from src.dataset import BrazilianDocumentDataset, get_transforms
from src.model import create_model


def train():
    train_path = Path(TRAIN_DIR)
    if not train_path.exists() or not list(train_path.iterdir()):
        print("No training data found. Run split_data.py first after placing data in data/raw/")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = BrazilianDocumentDataset(
        train_path,
        transform=get_transforms(IMG_SIZE, is_train=True),
    )
    if len(dataset) == 0:
        print("No images found in training directory.")
        return

    num_classes = len(dataset.class_to_idx)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

    model = create_model(num_classes=num_classes, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nTraining on {len(dataset)} samples, {num_classes} classes")
    print(f"Classes: {list(dataset.class_to_idx.keys())}\n")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / len(dataloader)
        acc = 100.0 * correct / total
        scheduler.step(avg_loss)

        print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {avg_loss:.4f} | Train Acc: {acc:.2f}%")

    torch.save({
        "model_state": model.state_dict(),
        "class_to_idx": dataset.class_to_idx,
        "num_classes": num_classes,
    }, MODEL_SAVE_DIR / "document_classifier.pt")
    print(f"\nModel saved to {MODEL_SAVE_DIR / 'document_classifier.pt'}")


if __name__ == "__main__":
    train()
