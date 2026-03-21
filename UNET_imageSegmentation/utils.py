import os
import torch
import torchvision
from torch.utils.data import DataLoader

from dataset import CarvanaDataset


# =============================================================
# Save and Load Checkpoints
# =============================================================

def save_checkpoint(state, filename: str = "my_checkpoint.pth.tar"):
    """
    Save model checkpoint

    Args:
        state (dict): model state dict + optimizer state dict
        filename (str): file name to save checkpoint
    """
    print("=> Saving checkpoint...")
    torch.save(state, filename)



def load_checkpoint(checkpoint, model):
    """
    Load model weights from checkpoint

    Args:
        checkpoint (dict): loaded checkpoint
        model (torch.nn.Module): model where weights will be loaded
    """
    print("=> Loading checkpoint...")
    model.load_state_dict(checkpoint["state_dict"])


# =============================================================
# DataLoader Function (CPU friendly + GPU ready)
# =============================================================

def get_loaders(
    train_dir: str,
    train_maskdir: str,
    val_dir: str,
    val_maskdir: str,
    batch_size: int,
    train_transform,
    val_transform,
    num_workers: int = 2,
    pin_memory: bool = True,
):
    """
    Create training and validation dataloaders
    """

    # --------------------------
    # Training Dataset
    # --------------------------
    train_ds = CarvanaDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    # Training DataLoader
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,   # Increase if CPU is strong
        pin_memory=pin_memory,     # Important for GPU training
    )

    # --------------------------
    # Validation Dataset
    # --------------------------
    val_ds = CarvanaDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    # Validation DataLoader
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader


# =============================================================
# Model Evaluation (Accuracy + Dice Score)
# =============================================================

def check_accuracy(loader, model, device: str = "cuda"):
    """
    Evaluate model performance using pixel accuracy and Dice score
    """

    num_correct = 0
    num_pixels = 0
    dice_score = 0

    # Put model in evaluation mode
    model.eval()

    with torch.no_grad():
        for x, y in loader:

            # Move data to GPU (if available)
            x = x.to(device)
            y = y.to(device).unsqueeze(1)

            # Model prediction
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

            # Pixel accuracy
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)

            # Dice score (important for segmentation)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(f"Got {num_correct}/{num_pixels} with accuracy {num_correct/num_pixels*100:.2f}%")
    print(f"Dice score: {dice_score/len(loader)}")

    # Switch back to training mode
    model.train()


# =============================================================
# Save Predictions as Images
# =============================================================

def save_predictions_as_imgs(
    loader,
    model,
    folder: str = "saved_images/",
    device: str = "cuda",
):
    """
    Save predicted masks and ground truth masks as PNG images
    """

    # Create folder if it does not exist
    os.makedirs(folder, exist_ok=True)

    model.eval()

    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)

        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

        # Save predicted mask
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )

        # Save ground truth mask
        torchvision.utils.save_image(
            y.unsqueeze(1), f"{folder}/gt_{idx}.png"
        )

    model.train()
