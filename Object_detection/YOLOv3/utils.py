"""
YOLOv3 Utility Functions

Includes:
- IoU functions
- Non-Max Suppression
- mAP calculation
- Bounding box conversions
- Visualization
- DataLoader helpers
- Checkpoint utilities

Optimized for:
- CPU training (low RAM laptops)
- GPU training (fast inference)
"""

import config
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import os

from collections import Counter
from tqdm import tqdm
from torch.utils.data import DataLoader


# =========================================================
# IoU between width & height only (used for anchor matching)
# =========================================================
def iou_width_height(boxes1, boxes2):
    """
    boxes1: (N, 2) -> width, height
    boxes2: (N, 2) -> width, height
    """

    intersection = torch.min(boxes1[..., 0], boxes2[..., 0]) * \
                   torch.min(boxes1[..., 1], boxes2[..., 1])

    union = (
        boxes1[..., 0] * boxes1[..., 1] +
        boxes2[..., 0] * boxes2[..., 1] -
        intersection
    )

    return intersection / (union + 1e-6)


# =========================================================
# IoU between two bounding boxes
# =========================================================
def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Computes IoU between predicted boxes and ground truth boxes
    """

    if box_format == "midpoint":
        # Convert (x, y, w, h) -> (x1, y1, x2, y2)
        box1_x1 = boxes_preds[..., 0] - boxes_preds[..., 2] / 2
        box1_y1 = boxes_preds[..., 1] - boxes_preds[..., 3] / 2
        box1_x2 = boxes_preds[..., 0] + boxes_preds[..., 2] / 2
        box1_y2 = boxes_preds[..., 1] + boxes_preds[..., 3] / 2

        box2_x1 = boxes_labels[..., 0] - boxes_labels[..., 2] / 2
        box2_y1 = boxes_labels[..., 1] - boxes_labels[..., 3] / 2
        box2_x2 = boxes_labels[..., 0] + boxes_labels[..., 2] / 2
        box2_y2 = boxes_labels[..., 1] + boxes_labels[..., 3] / 2

    else:  # corners format
        box1_x1, box1_y1, box1_x2, box1_y2 = boxes_preds.unbind(-1)
        box2_x1, box2_y1, box2_x2, box2_y2 = boxes_labels.unbind(-1)

    # Intersection area
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    # Union area
    box1_area = torch.abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = torch.abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


# =========================================================
# Non-Max Suppression
# =========================================================
def non_max_suppression(bboxes, iou_threshold, threshold, box_format="corners"):
    """
    Removes overlapping bounding boxes
    """

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)

    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0] or
               intersection_over_union(
                   torch.tensor(chosen_box[2:]),
                   torch.tensor(box[2:]),
                   box_format=box_format,
               ) < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


# =========================================================
# Mean Average Precision (mAP)
# =========================================================
def mean_average_precision(
    pred_boxes,
    true_boxes,
    iou_threshold=0.5,
    box_format="midpoint",
    num_classes=20
):
    """
    Calculates mAP across all classes
    """

    average_precisions = []
    epsilon = 1e-6

    for c in range(num_classes):

        detections = [d for d in pred_boxes if d[1] == c]
        ground_truths = [g for g in true_boxes if g[1] == c]

        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        detections.sort(key=lambda x: x[2], reverse=True)

        TP = torch.zeros(len(detections))
        FP = torch.zeros(len(detections))

        total_true_bboxes = len(ground_truths)
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):

            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)

        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)

        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))

        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)


# =========================================================
# Plot Bounding Boxes
# =========================================================
def plot_image(image, boxes):
    """
    Draw bounding boxes on image
    """

    cmap = plt.get_cmap("tab20b")
    class_labels = (
        config.COCO_LABELS if config.DATASET == "COCO"
        else config.PASCAL_CLASSES
    )

    colors = [cmap(i) for i in np.linspace(0, 1, len(class_labels))]

    im = np.array(image)
    height, width, _ = im.shape

    fig, ax = plt.subplots(1)
    ax.imshow(im)

    for box in boxes:

        class_pred = box[0]
        box = box[2:]

        x = box[0] - box[2] / 2
        y = box[1] - box[3] / 2

        rect = patches.Rectangle(
            (x * width, y * height),
            box[2] * width,
            box[3] * height,
            linewidth=2,
            edgecolor=colors[int(class_pred)],
            facecolor="none",
        )

        ax.add_patch(rect)

        plt.text(
            x * width,
            y * height,
            class_labels[int(class_pred)],
            color="white",
            bbox={"color": colors[int(class_pred)], "pad": 0},
        )

    plt.show()


# =========================================================
# Seed Everything (Reproducibility)
# =========================================================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False