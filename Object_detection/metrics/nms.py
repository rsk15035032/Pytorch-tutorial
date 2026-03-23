import torch
from IoU import intersection_over_union


def non_max_suppression(bboxes, iou_threshold, score_threshold, box_format="corners"):
    """
    Applies Non-Max Suppression (NMS) to remove overlapping bounding boxes.

    Args:
        bboxes (list): List of bounding boxes in the format
                       [class_id, confidence_score, x1, y1, x2, y2]
        iou_threshold (float): IoU threshold used to suppress overlapping boxes
        score_threshold (float): Remove boxes with confidence below this value
        box_format (str): "corners" or "midpoint" format of the bounding boxes

    Returns:
        list: Bounding boxes remaining after NMS
    """

    # Ensure input is a list (common mistake during training)
    assert isinstance(bboxes, list), "bboxes should be a list"

    # Step 1: Remove low-confidence boxes
    bboxes = [box for box in bboxes if box[1] > score_threshold]

    # Step 2: Sort boxes by confidence score (highest first)
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)

    bboxes_after_nms = []

    # Step 3: Iterate while there are still boxes left
    while bboxes:

        # Always select the box with the highest confidence
        chosen_box = bboxes.pop(0)

        # Compare this box with the remaining boxes
        bboxes = [
            box
            for box in bboxes
            # Keep the box if:
            # 1. It belongs to a different class OR
            # 2. IoU is smaller than the threshold
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            ) < iou_threshold
        ]

        # Save the chosen box
        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms

print(f'All right!')