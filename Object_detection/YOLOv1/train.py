"""
Train YOLOv1 on Pascal VOC Dataset

Optimized version:
- CPU friendly
- GPU ready
- Clean training loop
- Minimal but clear comments
"""

import torch
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader

from model import Yolov1
from dataset import VOCDataset
from loss import YoloLoss
from utils import (
    get_bboxes,
    mean_average_precision,
    load_checkpoint,
)

# -------------------------------
# Reproducibility
# -------------------------------
seed = 123
torch.manual_seed(seed)

# -------------------------------
# Hyperparameters
# -------------------------------
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8             # safer for low-RAM laptops
WEIGHT_DECAY = 0
EPOCHS = 100
NUM_WORKERS = 0            # important for CPU-only laptops
PIN_MEMORY = True if DEVICE == "cuda" else False

LOAD_MODEL = False
LOAD_MODEL_FILE = "overfit.pth.tar"

IMG_DIR = "data/images"
LABEL_DIR = "data/labels"

# -------------------------------
# Image Transform
# -------------------------------
class Compose:
    """Apply transforms while keeping bounding boxes unchanged"""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img = t(img)
        return img, bboxes


transform = Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
])

# -------------------------------
# Training Function
# -------------------------------
def train_fn(train_loader, model, optimizer, loss_fn):
    """One full training epoch"""

    model.train()
    loop = tqdm(train_loader, leave=True)

    losses = []

    for images, targets in loop:

        images = images.to(DEVICE)
        targets = targets.to(DEVICE)

        # Forward
        predictions = model(images)
        loss = loss_fn(predictions, targets)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        loop.set_postfix(loss=loss.item())

    print(f"Mean Training Loss: {sum(losses)/len(losses):.4f}")


# -------------------------------
# Main Training Script
# -------------------------------
def main():

    # Model
    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # Loss
    loss_fn = YoloLoss()

    # Load pretrained model if required
    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

    # -------------------------------
    # Dataset
    # -------------------------------
    train_dataset = VOCDataset(
        csv_file="data/100examples.csv",
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
    )

    test_dataset = VOCDataset(
        csv_file="data/test.csv",
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
    )

    # -------------------------------
    # DataLoader
    # -------------------------------
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False,
        drop_last=False,
    )

    # -------------------------------
    # Training Loop
    # -------------------------------
    for epoch in range(EPOCHS):

        print(f"\nEpoch [{epoch+1}/{EPOCHS}]")

        # Train
        train_fn(train_loader, model, optimizer, loss_fn)

        # -------------------------------
        # Evaluate using mAP
        # -------------------------------
        model.eval()

        pred_boxes, target_boxes = get_bboxes(
            test_loader,
            model,
            iou_threshold=0.5,
            threshold=0.4,
            device=DEVICE,
        )

        mean_avg_prec = mean_average_precision(
            pred_boxes,
            target_boxes,
            iou_threshold=0.5,
            box_format="midpoint",
        )

        print(f"Validation mAP: {mean_avg_prec:.4f}")


# -------------------------------
# Entry Point
# -------------------------------
if __name__ == "__main__":
    main()


import os
print(os.path.exists("data/100examples.csv"))