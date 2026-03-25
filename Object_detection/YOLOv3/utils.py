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
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import random
import torch

from collections import Counter
from torch.utils.data import DataLoader
from tqdm import tqdm


# ============================================================
#  DEVICE SETUP (CPU friendly + GPU ready)
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
#  IoU for width & height (used for anchor box matching)
# ============================================================
def iou_width_height(boxes1, boxes2):
    """
    Computes IoU only using width and height.
    Used while assigning anchor boxes.
    """

    intersection = torch.min(boxes1[..., 0], boxes2[..., 0]) * torch.min(
        boxes1[..., 1], boxes2[..., 1]
    )

    union = (
        boxes1[..., 0] * boxes1[..., 1] +
        boxes2[..., 0] * boxes2[..., 1] -
        intersection
    )

    return intersection / union


# ============================================================
#  FULL IoU calculation
# ============================================================
def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Calculates IoU between prediction and ground truth boxes.
    """

    if box_format == "midpoint":
        # Convert (x,y,w,h) -> (x1,y1,x2,y2)
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2

        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    else:
        # Already in (x1,y1,x2,y2)
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]

        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


# ============================================================
#  NON MAX SUPPRESSION (removes duplicate boxes)
# ============================================================
def non_max_suppression(bboxes, iou_threshold, threshold, box_format="corners"):

    assert type(bboxes) == list

    # Remove low confidence boxes
    bboxes = [box for box in bboxes if box[1] > threshold]

    # Sort by confidence score
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)

    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box for box in bboxes
            if box[0] != chosen_box[0] or
            intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            ) < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


# ============================================================
#  MEAN AVERAGE PRECISION (mAP)
# ============================================================
def mean_average_precision(
    pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=20
):

    average_precisions = []
    epsilon = 1e-6

    for c in range(num_classes):

        detections = []
        ground_truths = []

        # Collect boxes of class c
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

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


# ============================================================
#  GET EVALUATION BBOXES
# ============================================================

def get_evaluation_bboxes(
    loader,
    model,
    iou_threshold,
    anchors,
    threshold,
    box_format="midpoint",
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    """
    This function collects predicted bounding boxes and true boxes
    so that we can calculate mAP.
    """

    model.eval()
    train_idx = 0

    all_pred_boxes = []
    all_true_boxes = []

    for batch_idx, (x, labels) in enumerate(tqdm(loader)):
        x = x.to(device)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        bboxes = [[] for _ in range(batch_size)]

        # predictions come from 3 scales
        for i in range(3):
            S = predictions[i].shape[2]

            anchor = torch.tensor(anchors[i]).to(device) * S

            boxes_scale_i = cells_to_bboxes(
                predictions[i], anchor, S=S, is_preds=True
            )

            for idx, box in enumerate(boxes_scale_i):
                bboxes[idx] += box

        # true boxes (only from last scale)
        true_bboxes = cells_to_bboxes(
            labels[2].to(device), anchor, S=S, is_preds=False
        )

        for idx in range(batch_size):

            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format,
            )

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()

    return all_pred_boxes, all_true_boxes


# ============================================================
#  CELL → BOUNDING BOX CONVERSION (VERY IMPORTANT)
# ============================================================
def cells_to_bboxes(predictions, anchors, S, is_preds=True):

    BATCH_SIZE = predictions.shape[0]
    num_anchors = len(anchors)

    box_predictions = predictions[..., 1:5]

    if is_preds:
        anchors = anchors.reshape(1, len(anchors), 1, 1, 2).to(predictions.device)

        box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])
        box_predictions[..., 2:] = torch.exp(box_predictions[..., 2:]) * anchors

        scores = torch.sigmoid(predictions[..., 0:1])
        best_class = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1)

    else:
        scores = predictions[..., 0:1]
        best_class = predictions[..., 5:6]

    cell_indices = (
        torch.arange(S)
        .repeat(BATCH_SIZE, 3, S, 1)
        .unsqueeze(-1)
        .to(predictions.device)
    )

    x = (box_predictions[..., 0:1] + cell_indices) / S
    y = (box_predictions[..., 1:2] + cell_indices.permute(0,1,3,2,4)) / S
    w_h = box_predictions[..., 2:4] / S

    converted_bboxes = torch.cat((best_class, scores, x, y, w_h), dim=-1)

    return converted_bboxes.reshape(BATCH_SIZE, num_anchors * S * S, 6).tolist()


# ============================================================
#  CHECK CLASS ACCURACY
# ============================================================
def check_class_accuracy(model, loader, threshold):

    model.eval()

    tot_class_preds, correct_class = 0, 0
    tot_noobj, correct_noobj = 0, 0
    tot_obj, correct_obj = 0, 0

    for idx, (x, y) in enumerate(tqdm(loader)):

        x = x.to(DEVICE)

        with torch.no_grad():
            out = model(x)

        for i in range(3):

            y[i] = y[i].to(DEVICE)

            obj = y[i][..., 0] == 1
            noobj = y[i][..., 0] == 0

            correct_class += torch.sum(
                torch.argmax(out[i][..., 5:][obj], dim=-1) == y[i][..., 5][obj]
            )

            tot_class_preds += torch.sum(obj)

            obj_preds = torch.sigmoid(out[i][..., 0]) > threshold

            correct_obj += torch.sum(obj_preds[obj] == y[i][..., 0][obj])
            tot_obj += torch.sum(obj)

            correct_noobj += torch.sum(obj_preds[noobj] == y[i][..., 0][noobj])
            tot_noobj += torch.sum(noobj)

    print(f"Class accuracy: {(correct_class/(tot_class_preds+1e-16))*100:.2f}%")
    print(f"No-obj accuracy: {(correct_noobj/(tot_noobj+1e-16))*100:.2f}%")
    print(f"Obj accuracy: {(correct_obj/(tot_obj+1e-16))*100:.2f}%")

    model.train()


# ============================================================
#  SAVE / LOAD CHECKPOINT
# ============================================================
def save_checkpoint(model, optimizer, filename="checkpoint.pth.tar"):
    print("Saving checkpoint...")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_file, map_location=DEVICE)

    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


# ============================================================
#  DATALOADER SETUP
# ============================================================
def get_loaders(train_csv_path, test_csv_path):
    from dataset import YOLODataset

    IMAGE_SIZE = config.IMAGE_SIZE

    train_dataset = YOLODataset(
        train_csv_path,
        config.IMG_DIR,
        config.LABEL_DIR,
        config.ANCHORS,
        image_size=IMAGE_SIZE,
        S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
        C=config.NUM_CLASSES,
        transform=config.train_transforms,
    )

    test_dataset = YOLODataset(
        test_csv_path,
        config.IMG_DIR,
        config.LABEL_DIR,
        config.ANCHORS,
        image_size=IMAGE_SIZE,
        S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
        C=config.NUM_CLASSES,
        transform=config.test_transforms,
    )

    train_eval_dataset = YOLODataset(
        train_csv_path,
        config.IMG_DIR,
        config.LABEL_DIR,
        config.ANCHORS,
        image_size=IMAGE_SIZE,
        S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
        C=config.NUM_CLASSES,
        transform=config.test_transforms,
    )

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    train_eval_loader = DataLoader(train_eval_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    return train_loader, test_loader, train_eval_loader


# ============================================================
#  FIX RANDOMNESS (important for reproducibility)
# ============================================================
def seed_everything(seed=42):

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False