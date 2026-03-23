"""
YOLOv1 Loss Function

Implements the loss described in the original YOLO paper.
Components:
1. Localization Loss (bounding box coordinates)
2. Objectness Loss
3. No-object Loss
4. Class Probability Loss
"""

import torch
import torch.nn as nn
from utils import intersection_over_union


class YoloLoss(nn.Module):
    """YOLOv1 Loss"""

    def __init__(self, S=7, B=2, C=20):
        super().__init__()

        self.mse = nn.MSELoss(reduction="sum")

        # Grid size, number of boxes, number of classes
        self.S = S
        self.B = B
        self.C = C

        # Loss weights from YOLO paper
        self.lambda_coord = 5
        self.lambda_noobj = 0.5

    def forward(self, predictions, target):
        """
        predictions shape: (batch_size, S*S*(C + B*5))
        target shape: same as predictions
        """

        # Reshape predictions to (BATCH, S, S, C + B*5)
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)

        # -------------------------------------------------
        # IoU of predicted boxes with target box
        # -------------------------------------------------
        iou_b1 = intersection_over_union(
            predictions[..., 21:25], target[..., 21:25]
        )
        iou_b2 = intersection_over_union(
            predictions[..., 26:30], target[..., 21:25]
        )

        ious = torch.stack([iou_b1, iou_b2], dim=0)

        # Select the best bounding box prediction
        iou_maxes, best_box = torch.max(ious, dim=0)

        # Indicator if object exists in cell
        exists_box = target[..., 20].unsqueeze(3)

        # -------------------------------------------------
        # 1. Localization Loss (x, y, w, h)
        # -------------------------------------------------
        box_predictions = exists_box * (
            best_box * predictions[..., 26:30]
            + (1 - best_box) * predictions[..., 21:25]
        )

        box_targets = exists_box * target[..., 21:25]

        # sqrt on width & height (as described in YOLO paper)
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )

        # -------------------------------------------------
        # 2. Objectness Loss
        # -------------------------------------------------
        pred_box = (
            best_box * predictions[..., 25:26]
            + (1 - best_box) * predictions[..., 20:21]
        )

        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 20:21]),
        )

        # -------------------------------------------------
        # 3. No Object Loss
        # -------------------------------------------------
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
        )

        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
        )

        # -------------------------------------------------
        # 4. Classification Loss
        # -------------------------------------------------
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :20], end_dim=-2),
            torch.flatten(exists_box * target[..., :20], end_dim=-2),
        )

        # -------------------------------------------------
        # Total YOLO Loss
        # -------------------------------------------------
        loss = (
            self.lambda_coord * box_loss
            + object_loss
            + self.lambda_noobj * no_object_loss
            + class_loss
        )

        return loss