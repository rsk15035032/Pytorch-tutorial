"""
PyTorch Dataset for Pascal VOC (YOLOv1 format)

Reads:
- image path from CSV
- label file in YOLO format: class x y w h

Returns:
image tensor + YOLO label matrix (S x S x (C + 5B))
"""

import os
import torch
import pandas as pd
from PIL import Image


class VOCDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        csv_file,
        img_dir,
        label_dir,
        S=7,
        B=2,
        C=20,
        transform=None,
    ):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform

        # YOLO parameters
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):

        # -------------------------------
        # Load labels
        # -------------------------------
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])

        boxes = []
        with open(label_path) as f:
            for line in f.readlines():
                class_label, x, y, w, h = map(float, line.strip().split())
                boxes.append([class_label, x, y, w, h])

        boxes = torch.tensor(boxes)

        # -------------------------------
        # Load image
        # -------------------------------
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path).convert("RGB")

        # -------------------------------
        # Apply transforms
        # -------------------------------
        if self.transform:
            image, boxes = self.transform(image, boxes)

        # -------------------------------
        # Convert to YOLO label matrix
        # -------------------------------
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))

        for box in boxes:
            class_label, x, y, w, h = box.tolist()
            class_label = int(class_label)

            # Determine which grid cell the object belongs to
            i, j = int(self.S * y), int(self.S * x)

            # Coordinates relative to the cell
            x_cell = self.S * x - j
            y_cell = self.S * y - i

            # Width & height relative to the cell
            w_cell = w * self.S
            h_cell = h * self.S

            # Only one object per cell (YOLOv1 limitation)
            if label_matrix[i, j, 20] == 0:
                label_matrix[i, j, 20] = 1

                label_matrix[i, j, 21:25] = torch.tensor(
                    [x_cell, y_cell, w_cell, h_cell]
                )

                # One-hot class encoding
                label_matrix[i, j, class_label] = 1

        return image, label_matrix