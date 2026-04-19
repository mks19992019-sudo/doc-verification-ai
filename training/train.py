"""
CNN Training Script for Document Forgery Detection
Fine-tunes ResNet-50 on genuine vs forged documents.

Expected dataset structure:
    dataset/
        train/
            genuine/    ← genuine document images
            forged/     ← forged/manipulated document images
        val/
            genuine/
            forged/

Usage:
    python train.py --epochs 10 --batch_size 16 --lr 1e-4
"""

import os
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm


class DocumentDataset(Dataset):
    def __init__(self, root: str, split: str = "train", img_size: int = 224):
        self.split = split
        self.samples = []
        self.img_size = img_size

        split_root = Path(root) / split
        for label, name in enumerate(["genuine", "forged"]):
            class_dir = split_root / name
            if not class_dir.exists():
                raise FileNotFoundError(f"Class directory not found: {class_dir}")
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".tiff", ".bmp"]:
                    self.samples.append((str(img_path), label))

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, label, img_path


def build_model(device: torch.device) -> nn.Module:
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False

    for param in model.layer4.parameters():
        param.requires_grad = True

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_features, 1),
    )
    return model.to(device)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels, _ in tqdm(loader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5).int().cpu().numpy()
        all_preds.extend(preds.flatten())
        all_labels.extend(labels.cpu().numpy().flatten())

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )
    return running_loss / len(loader), acc, f1


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels, _ in tqdm(loader, desc="Validating", leave=False):
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).int().cpu().numpy()
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )
    return running_loss / len(loader), acc, f1


def main():
    parser = argparse.ArgumentParser(description="Train ResNet-50 for document forgery detection")
    parser.add_argument("--data", type=str, default="dataset", help="Root dataset directory")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weights", type=str, default="weights/resnet50_forgery.pt", help="Output weights path")
    parser.add_argument("--img_size", type=int, default=224)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(os.path.dirname(args.weights), exist_ok=True)

    print("Loading datasets...")
    train_dataset = DocumentDataset(args.data, split="train", img_size=args.img_size)
    val_dataset = DocumentDataset(args.data, split="val", img_size=args.img_size)
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = build_model(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1 = 0.0
    best_model_state = None

    print(f"\nStarting training for {args.epochs} epochs...\n")
    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        print(f"Epoch {epoch+1}/{args.epochs} | {elapsed:.1f}s")
        print(f"  Train — loss: {train_loss:.4f}, acc: {train_acc:.4f}, f1: {train_f1:.4f}")
        print(f"  Val   — loss: {val_loss:.4f}, acc: {val_acc:.4f}, f1: {val_f1:.4f}")

        if val_f1 >= best_f1:
            best_f1 = val_f1
            best_model_state = model.state_dict().copy()
            print(f"  >> New best model saved (f1: {best_f1:.4f})")

        scheduler.step()

    if best_model_state is not None:
        torch.save(best_model_state, args.weights)
        print(f"\nTraining complete. Best weights saved to: {args.weights}")
    else:
        torch.save(model.state_dict(), args.weights)
        print(f"\nTraining complete. Weights saved to: {args.weights}")


if __name__ == "__main__":
    main()
