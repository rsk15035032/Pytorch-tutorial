"""
YOLOv3 Configuration File

Optimized for:
- CPU training (low RAM laptops)
- GPU training (fast dataloading)
- Clean configuration for production projects
"""

import albumentations as A
import cv2
import torch
from albumentations.pytorch import ToTensorV2
from utils import seed_everything


# =========================================================
# GENERAL SETTINGS
# =========================================================
DATASET = "Object_detection/YOLOv3/data"        # Change to COCO if needed
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Uncomment if you want reproducible results
# seed_everything(42)

NUM_WORKERS = 2                # Better for CPU training
BATCH_SIZE = 8                 # Safe for low-RAM laptops
IMAGE_SIZE = 416
NUM_CLASSES = 20               # 20 for VOC, 80 for COCO

LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 100

CONF_THRESHOLD = 0.05
MAP_IOU_THRESH = 0.5
NMS_IOU_THRESH = 0.45

# Grid sizes for YOLO scales
S = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]

PIN_MEMORY = True if DEVICE == "cuda" else False

LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_FILE = "checkpoint.pth.tar"

IMG_DIR = DATASET + "/images/"
LABEL_DIR = DATASET + "/labels/"


# =========================================================
# ANCHOR BOXES (scaled between 0 and 1)
# =========================================================
ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],   # large objects
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],  # medium objects
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],  # small objects
]


# =========================================================
# DATA AUGMENTATIONS (TRAIN)
# =========================================================
scale = 1.1

train_transforms = A.Compose(
    [
        # Resize image keeping aspect ratio
        A.LongestMaxSize(max_size=int(IMAGE_SIZE * scale)),

        # Pad image to square
        A.PadIfNeeded(
            min_height=int(IMAGE_SIZE * scale),
            min_width=int(IMAGE_SIZE * scale),
            border_mode=cv2.BORDER_CONSTANT,
        ),

        # Final random crop to 416x416
        A.RandomCrop(width=IMAGE_SIZE, height=IMAGE_SIZE),

        # Color augmentations
        A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.4),

        # Geometric transformations
        A.OneOf(
            [
                A.ShiftScaleRotate(rotate_limit=20, border_mode=cv2.BORDER_CONSTANT, p=0.5),
                A.Affine(shear=15, mode=cv2.BORDER_CONSTANT, p=0.5),
            ],
            p=1.0,
        ),

        # Flips
        A.HorizontalFlip(p=0.5),

        # Noise / small augmentations
        A.Blur(p=0.1),
        A.CLAHE(p=0.1),
        A.Posterize(p=0.1),
        A.ToGray(p=0.1),
        A.ChannelShuffle(p=0.05),

        # Normalize image
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255),

        # Convert image to PyTorch tensor
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),
)


# =========================================================
# DATA AUGMENTATIONS (TEST / VALIDATION)
# =========================================================
test_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=IMAGE_SIZE),

        A.PadIfNeeded(
            min_height=IMAGE_SIZE,
            min_width=IMAGE_SIZE,
            border_mode=cv2.BORDER_CONSTANT,
        ),

        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255),

        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),
)


# =========================================================
# CLASS LABELS
# =========================================================
PASCAL_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]


COCO_LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird",
    "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
    "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]