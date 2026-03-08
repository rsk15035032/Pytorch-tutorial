import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path

# =========================================================
# DEVICE CONFIGURATION
# =========================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_CUDA = torch.cuda.is_available()

# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path("data")
TRAIN_DIR = BASE_DIR / "train"
VAL_DIR = BASE_DIR / "val"

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_DISC = CHECKPOINT_DIR / "disc.pth.tar"
CHECKPOINT_GEN = CHECKPOINT_DIR / "gen.pth.tar"

# =========================================================
# TRAINING HYPERPARAMETERS
# =========================================================

IMAGE_SIZE = 256
CHANNELS_IMG = 3

# CPU-friendly defaults (auto-adjust)
BATCH_SIZE = 16 if USE_CUDA else 4
NUM_WORKERS = 2 if USE_CUDA else 0

LEARNING_RATE = 2e-4
NUM_EPOCHS = 20

L1_LAMBDA = 100
LAMBDA_GP = 10

LOAD_MODEL = False
SAVE_MODEL = True

# =========================================================
# OPTIMIZATION SETTINGS
# =========================================================

BETAS = (0.5, 0.999)

# =========================================================
# DATA AUGMENTATIONS
# =========================================================

both_transform = A.Compose(
    [
        A.Resize(width=IMAGE_SIZE, height=IMAGE_SIZE),
    ],
    additional_targets={"image0": "image"},
)

transform_only_input = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(p=0.2),
        A.Normalize(
            mean=[0.5] * CHANNELS_IMG,
            std=[0.5] * CHANNELS_IMG,
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)

transform_only_mask = A.Compose(
    [
        A.Normalize(
            mean=[0.5] * CHANNELS_IMG,
            std=[0.5] * CHANNELS_IMG,
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)

# =========================================================
# REPRODUCIBILITY
# =========================================================

SEED = 42
torch.manual_seed(SEED)

if USE_CUDA:
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.benchmark = True
else:
    torch.backends.cudnn.benchmark = False