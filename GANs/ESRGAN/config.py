import torch
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ==========================================================
# DEVICE CONFIGURATION
# ==========================================================

# Default: CPU (recommended for this repository)
DEVICE = "cpu"

# Uncomment below line to enable GPU training
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==========================================================
# MODEL CHECKPOINTS
# ==========================================================

LOAD_MODEL = False
SAVE_MODEL = True

CHECKPOINT_GEN = "gen.pth"
CHECKPOINT_DISC = "disc.pth"


# ==========================================================
# TRAINING HYPERPARAMETERS
# ==========================================================

LEARNING_RATE = 1e-4
NUM_EPOCHS = 10000

# CPU friendly batch size
BATCH_SIZE = 4

# If using GPU you can increase batch size
# BATCH_SIZE = 16

LAMBDA_GP = 10

# Number of workers for dataloader
NUM_WORKERS = 0   # CPU safe

# For GPU training you can increase workers
# NUM_WORKERS = 4


# ==========================================================
# IMAGE SETTINGS
# ==========================================================

HIGH_RES = 128
LOW_RES = HIGH_RES // 4
IMG_CHANNELS = 3


# ==========================================================
# TRANSFORMATIONS
# ==========================================================

# High Resolution Transform
highres_transform = A.Compose(
    [
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ToTensorV2(),
    ]
)

# Low Resolution Transform
lowres_transform = A.Compose(
    [
        A.Resize(width=LOW_RES, height=LOW_RES, interpolation=Image.BICUBIC),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ToTensorV2(),
    ]
)

# Data Augmentation applied to both images
both_transforms = A.Compose(
    [
        A.RandomCrop(width=HIGH_RES, height=HIGH_RES),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ]
)

# Test Transform
test_transform = A.Compose(
    [
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ToTensorV2(),
    ]
)