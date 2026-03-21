import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

from model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)


# =============================================================
# Hyperparameters (CPU friendly + GPU ready)
# =============================================================

LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 3
NUM_WORKERS = 2
IMAGE_HEIGHT = 160   # Reduced from 1280 for faster training
IMAGE_WIDTH = 240    # Reduced from 1918 for faster training
PIN_MEMORY = True
LOAD_MODEL = False

TRAIN_IMG_DIR = "data/train_images/"
TRAIN_MASK_DIR = "data/train_masks/"
VAL_IMG_DIR = "data/val_images/"
VAL_MASK_DIR = "data/val_masks/"


# =============================================================
# Training Function
# =============================================================

def train_fn(loader, model, optimizer, loss_fn, scaler):
    """
    One epoch training loop
    """

    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):

        # Move data to GPU (if available)
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # ---------------- Forward pass ----------------
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # ---------------- Backward pass ----------------
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update progress bar
        loop.set_postfix(loss=loss.item())


# =============================================================
# Main Training Pipeline
# =============================================================

def main():

    # -------------------------------------------------
    # Data Augmentation (Albumentations)
    # -------------------------------------------------
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    # -------------------------------------------------
    # Model + Loss + Optimizer
    # -------------------------------------------------
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)

    loss_fn = nn.BCEWithLogitsLoss()  # Best for binary segmentation
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # -------------------------------------------------
    # Load Data
    # -------------------------------------------------
    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    # -------------------------------------------------
    # Load saved model (optional)
    # -------------------------------------------------
    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    # Check model before training
    check_accuracy(val_loader, model, device=DEVICE)

    # Mixed precision training (faster on GPU)
    scaler = torch.cuda.amp.GradScaler()

    # -------------------------------------------------
    # Training Loop
    # -------------------------------------------------
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")

        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # Save model after every epoch
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # Evaluate model
        check_accuracy(val_loader, model, device=DEVICE)

        # Save predicted masks
        save_predictions_as_imgs(
            val_loader, model, folder="saved_images/", device=DEVICE
        )


# =============================================================
# Entry Point
# =============================================================

if __name__ == "__main__":
    main()
