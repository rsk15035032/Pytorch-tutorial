import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ==============================
# DEVICE CONFIGURATION
# ==============================

DEVICE = "cpu"   # Force CPU training for stability


# ==============================
# DATASET PATHS
# ==============================

TRAIN_DIR = "data/train"
VAL_DIR = "data/val"


# ==============================
# TRAINING HYPERPARAMETERS
# ==============================

BATCH_SIZE = 1                # Standard for GAN training
LEARNING_RATE = 2e-4          # Typical GAN learning rate
NUM_EPOCHS = 20               # Increase for better results


# ==============================
# LOSS WEIGHTS (CycleGAN)
# ==============================

LAMBDA_IDENTITY = 0.5         # Helps preserve colors
LAMBDA_CYCLE = 10             # Cycle consistency loss


# ==============================
# DATALOADER SETTINGS
# ==============================

NUM_WORKERS = 0               # Best for Windows CPU
PIN_MEMORY = False            # Disable for CPU


# ==============================
# CHECKPOINT SETTINGS
# ==============================

LOAD_MODEL = False
SAVE_MODEL = True

CHECKPOINT_GEN_H = "gen_horse.pth.tar"
CHECKPOINT_GEN_Z = "gen_zebra.pth.tar"
CHECKPOINT_CRITIC_H = "disc_horse.pth.tar"
CHECKPOINT_CRITIC_Z = "disc_zebra.pth.tar"


# ==============================
# IMAGE TRANSFORMATIONS
# ==============================

IMAGE_SIZE = 256

transforms = A.Compose(
    [
        A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),

        # Normalize to [-1,1] for GANs using Tanh
        A.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
            max_pixel_value=255,
        ),

        ToTensorV2(),
    ],

    # Apply same augmentation to both images
    additional_targets={"image0": "image"},
)