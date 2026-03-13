
'''
| Setting    | GPU Version | CPU Version |
| ---------- | ----------- | ----------- |
| Device     | `"cuda"`    | `"cpu"`     |
| Batch Size | 16          | **4**       |
| Workers    | 4           | **0**       |
| Image Size | 96          | **64**      |
'''

import torch
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_GEN = "gen.pth.tar"
CHECKPOINT_DISC = "disc.pth.tar"

# Force CPU
DEVICE = "cpu"

LEARNING_RATE = 1e-4
NUM_EPOCHS = 100

# Smaller batch for CPU
BATCH_SIZE = 4

# CPU works better with small workers
NUM_WORKERS = 0

# Smaller resolution = faster training
HIGH_RES = 64
LOW_RES = HIGH_RES // 4

IMG_CHANNELS = 3


# High resolution transform
highres_transform = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2(),
    ]
)

# Low resolution transform
lowres_transform = A.Compose(
    [
        A.Resize(width=LOW_RES, height=LOW_RES, interpolation=Image.BICUBIC),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ToTensorV2(),
    ]
)

# Data augmentation
both_transforms = A.Compose(
    [
        A.RandomCrop(width=HIGH_RES, height=HIGH_RES),
        A.HorizontalFlip(p=0.5),
    ]
)


# test transform
test_transform = A.Compose(
    [
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ToTensorV2(),
    ]
)
