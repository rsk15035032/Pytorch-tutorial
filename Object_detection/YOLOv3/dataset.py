"""
YOLOv3 Dataset Loader (Pascal VOC / COCO format)

Optimized for:
- CPU training (low-RAM laptops)
- GPU training (fast tensor creation)
- Stable handling of empty label files
- Clean target generation for 3 detection scales
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile

import config
from utils import iou_width_height as iou

# Allows loading broken / partially saved images safely
ImageFile.LOAD_TRUNCATED_IMAGES = True


# =========================================================
# YOLO DATASET CLASS
# =========================================================
class YOLODataset(Dataset):
    """
    Dataset for YOLOv3

    Input:
        image → (3, 416, 416)
        targets → tuple of 3 tensors (for 3 scales)

    Output target shape per scale:
        (anchors_per_scale, S, S, 6)

        where 6 = [object, x, y, w, h, class]
    """

    def __init__(
        self,
        csv_file,
        img_dir,
        label_dir,
        anchors,
        image_size=416,
        S=[13, 26, 52],
        C=20,
        transform=None,
    ):
        super().__init__()

        # Load CSV (image path + label path)
        self.annotations = pd.read_csv(csv_file)

        self.img_dir = img_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.transform = transform

        # YOLO parameters
        self.S = S
        self.C = C
        self.ignore_iou_thresh = 0.5

        # Convert anchors into tensor
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3

    # =====================================================
    # Dataset length
    # =====================================================
    def __len__(self):
        return len(self.annotations)

    # =====================================================
    # Load one sample
    # =====================================================
    def __getitem__(self, index):

        # ---------- Load image ----------
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = np.array(Image.open(img_path).convert("RGB"))

        # ---------- Load labels ----------
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])

        # Safe loading (works even if label file is empty)
        if os.path.getsize(label_path) == 0:
            bboxes = []
        else:
            bboxes = np.loadtxt(label_path, delimiter=" ", ndmin=2).tolist()

        # YOLO format → (x, y, w, h, class)
        bboxes = [box[1:] + [box[0]] for box in bboxes]

        # ---------- Apply augmentations ----------
        if self.transform:
            transformed = self.transform(image=image, bboxes=bboxes)
            image = transformed["image"]
            bboxes = transformed["bboxes"]

        # =================================================
        # Create empty targets for 3 scales
        # =================================================
        targets = [
            torch.zeros((self.num_anchors_per_scale, S, S, 6))
            for S in self.S
        ]

        # =================================================
        # Assign ground-truth boxes to anchors
        # =================================================
        for box in bboxes:

            x, y, width, height, class_label = box

            # Compute IoU between GT box and all anchors
            iou_anchors = iou(torch.tensor([width, height]), self.anchors)

            # Sort anchors based on IoU
            anchor_indices = iou_anchors.argsort(descending=True)

            has_anchor = [False] * 3

            for anchor_idx in anchor_indices:

                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale

                S = self.S[scale_idx]

                # Find grid cell location
                i, j = int(S * y), int(S * x)

                # If anchor not used yet
                if targets[scale_idx][anchor_on_scale, i, j, 0] == 0 and not has_anchor[scale_idx]:

                    # Object exists
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1

                    # Convert to cell coordinates
                    x_cell = S * x - j
                    y_cell = S * y - i
                    width_cell = width * S
                    height_cell = height * S

                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell]
                    )

                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)

                    has_anchor[scale_idx] = True

                # Ignore prediction if IoU is high but anchor already used
                elif targets[scale_idx][anchor_on_scale, i, j, 0] == 0 and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1

        return image, tuple(targets)