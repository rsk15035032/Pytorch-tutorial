import torch


def intersection_over_union(pred_boxes, true_boxes, box_format="midpoint"):
    """
    Compute Intersection over Union (IoU) between predicted boxes and ground-truth boxes.

    Args:
        pred_boxes (Tensor): Predicted bounding boxes of shape (..., 4)
        true_boxes (Tensor): Ground truth bounding boxes of shape (..., 4)
        box_format (str): Format of the boxes
                          "midpoint" -> (x_center, y_center, width, height)
                          "corners"  -> (x1, y1, x2, y2)

    Returns:
        Tensor: IoU score for each box
    """

    # ------------------------------------------------------------
    # STEP 1: Convert boxes to corner format (x1, y1, x2, y2)
    # This makes intersection calculation easier
    # ------------------------------------------------------------

    if box_format == "midpoint":
        # Predicted box
        pred_x1 = pred_boxes[..., 0:1] - pred_boxes[..., 2:3] / 2
        pred_y1 = pred_boxes[..., 1:2] - pred_boxes[..., 3:4] / 2
        pred_x2 = pred_boxes[..., 0:1] + pred_boxes[..., 2:3] / 2
        pred_y2 = pred_boxes[..., 1:2] + pred_boxes[..., 3:4] / 2

        # Ground-truth box
        true_x1 = true_boxes[..., 0:1] - true_boxes[..., 2:3] / 2
        true_y1 = true_boxes[..., 1:2] - true_boxes[..., 3:4] / 2
        true_x2 = true_boxes[..., 0:1] + true_boxes[..., 2:3] / 2
        true_y2 = true_boxes[..., 1:2] + true_boxes[..., 3:4] / 2

    else:  # box_format == "corners"
        pred_x1 = pred_boxes[..., 0:1]
        pred_y1 = pred_boxes[..., 1:2]
        pred_x2 = pred_boxes[..., 2:3]
        pred_y2 = pred_boxes[..., 3:4]

        true_x1 = true_boxes[..., 0:1]
        true_y1 = true_boxes[..., 1:2]
        true_x2 = true_boxes[..., 2:3]
        true_y2 = true_boxes[..., 3:4]

    # ------------------------------------------------------------
    # STEP 2: Compute intersection box
    # The overlapping region is defined by:
    # left   = max(x1)
    # top    = max(y1)
    # right  = min(x2)
    # bottom = min(y2)
    # ------------------------------------------------------------

    x1 = torch.max(pred_x1, true_x1)
    y1 = torch.max(pred_y1, true_y1)
    x2 = torch.min(pred_x2, true_x2)
    y2 = torch.min(pred_y2, true_y2)

    # If boxes do not overlap, width/height becomes negative
    # clamp(0) makes intersection = 0 instead of negative
    intersection = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    # ------------------------------------------------------------
    # STEP 3: Compute area of both boxes
    # ------------------------------------------------------------

    pred_area = abs((pred_x2 - pred_x1) * (pred_y2 - pred_y1))
    true_area = abs((true_x2 - true_x1) * (true_y2 - true_y1))

    # ------------------------------------------------------------
    # STEP 4: Compute IoU using formula
    # IoU = Intersection / Union
    # Union = area1 + area2 - intersection
    # Small epsilon added to avoid division by zero
    # ------------------------------------------------------------

    union = pred_area + true_area - intersection

    iou = intersection / (union + 1e-6)

    return iou

print(f'Ok!')