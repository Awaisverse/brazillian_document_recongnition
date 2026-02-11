"""
Evaluate model on the held-out 40% test set (NOT used during training)
"""
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import BATCH_SIZE, IMG_SIZE, MODEL_SAVE_DIR, TEST_DIR
from src.dataset import BrazilianDocumentDataset, get_transforms
from src.model import create_model


def evaluate():
    test_path = Path(TEST_DIR)
    if not test_path.exists() or not list(test_path.iterdir()):
        print("No test data found. Run split_data.py first.")
        return

    ckpt_path = MODEL_SAVE_DIR / "document_classifier.pt"
    if not ckpt_path.exists():
        print("No trained model found. Run train.py first.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ckpt = torch.load(ckpt_path, map_location=device)
    num_classes = ckpt["num_classes"]
    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    model = create_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()

    dataset = BrazilianDocumentDataset(
        test_path,
        transform=get_transforms(IMG_SIZE, is_train=False),
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    correct = 0
    total = 0
    class_correct = {c: 0 for c in range(num_classes)}
    class_total = {c: 0 for c in range(num_classes)}

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            for l, p in zip(labels, predicted):
                class_total[l.item()] += 1
                if l.item() == p.item():
                    class_correct[l.item()] += 1

    acc = 100.0 * correct / total
    print("\n" + "=" * 50)
    print("Test Set Accuracy (40% held-out)")
    print("=" * 50)
    print(f"Overall Accuracy: {acc:.2f}% ({correct}/{total})")
    print("\nPer-class accuracy:")
    for idx in range(num_classes):
        cls_name = idx_to_class[idx]
        c_tot = class_total[idx]
        c_cor = class_correct[idx]
        c_acc = 100.0 * c_cor / c_tot if c_tot > 0 else 0
        print(f"  {cls_name}: {c_acc:.2f}% ({c_cor}/{c_tot})")
    print("=" * 50)


if __name__ == "__main__":
    evaluate()
