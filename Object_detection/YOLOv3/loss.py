"""
YOLOv3 Loss Function (PyTorch)

This implementation follows the YOLOv3 paper with one change:
Class loss uses CrossEntropy instead of Binary CrossEntropy.

Optimized for:
- CPU training (less memory usage)
- GPU training (no unnecessary tensor copies)
- Numerical stability
"""

import torch
import torch.nn as nn
from utils import intersection_over_union


class YoloLoss(nn.Module):
    """
    YOLO Loss = Box Loss + Object Loss + No Object Loss + Class Loss
    """

    def __init__(self):
        super().__init__()

        # Basic loss functions
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.ce = nn.CrossEntropyLoss()

        self.sigmoid = nn.Sigmoid()

        # Loss weights (same idea as YOLO paper)
        self.lambda_box = 10
        self.lambda_obj = 1
        self.lambda_noobj = 10
        self.lambda_class = 1

    # =========================================================
    # Forward Pass
    # =========================================================
    def forward(self, predictions, targets, anchors):
        """
        predictions shape:
        (BATCH_SIZE, 3, S, S, 5 + num_classes)

        targets shape:
        (BATCH_SIZE, 3, S, S, 5 + num_classes)

        anchors shape:
        (3, 2)
        """

        # =====================================================
        # Object / No-object masks
        # =====================================================
        obj = targets[..., 0] == 1       # where object exists
        noobj = targets[..., 0] == 0     # where no object exists

        # =====================================================
        # 1. NO OBJECT LOSS
        # Penalize confidence when object does not exist
        # =====================================================
        no_object_loss = self.bce(
            predictions[..., 0:1][noobj],
            targets[..., 0:1][noobj],
        )

        # =====================================================
        # 2. OBJECT LOSS
        # Encourage confidence = IoU when object exists
        # =====================================================
        anchors = anchors.reshape(1, 3, 1, 1, 2)

        # Predicted bounding boxes
        pred_boxes = torch.cat(
            [
                self.sigmoid(predictions[..., 1:3]),                    # x, y
                torch.exp(predictions[..., 3:5]) * anchors              # w, h
            ],
            dim=-1,
        )

        # Compute IoU between predicted boxes and target boxes
        ious = intersection_over_union(
            pred_boxes[obj],
            targets[..., 1:5][obj]
        ).detach()

        object_loss = self.mse(
            self.sigmoid(predictions[..., 0:1][obj]),
            ious * targets[..., 0:1][obj],
        )

        # =====================================================
        # 3. BOX COORDINATE LOSS
        # Train x, y, w, h only when object exists
        # =====================================================
        pred_xy = self.sigmoid(predictions[..., 1:3])
        pred_wh = predictions[..., 3:5]

        target_xy = targets[..., 1:3]
        target_wh = torch.log(
            1e-16 + targets[..., 3:5] / anchors
        )

        box_loss = self.mse(
            torch.cat([pred_xy, pred_wh], dim=-1)[obj],
            torch.cat([target_xy, target_wh], dim=-1)[obj],
        )

        # =====================================================
        # 4. CLASS LOSS
        # Predict correct class only where object exists
        # =====================================================
        class_loss = self.ce(
            predictions[..., 5:][obj],
            targets[..., 5][obj].long(),
        )

        # =====================================================
        # FINAL YOLO LOSS
        # =====================================================
        total_loss = (
            self.lambda_box * box_loss
            + self.lambda_obj * object_loss
            + self.lambda_noobj * no_object_loss
            + self.lambda_class * class_loss
        )

        return total_loss