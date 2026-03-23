import torch
from collections import Counter
from IoU import intersection_over_union


def mean_average_precision(
    pred_boxes,
    true_boxes,
    iou_threshold=0.5,
    box_format="midpoint",
    num_classes=20,
):
    """
    Computes Mean Average Precision (mAP) for object detection.

    Args:
        pred_boxes (list): Predictions in format
                           [image_idx, class_id, confidence, x1, y1, x2, y2]
        true_boxes (list): Ground truth boxes in same format as pred_boxes
        iou_threshold (float): Minimum IoU required to consider a prediction correct
        box_format (str): "midpoint" or "corners"
        num_classes (int): Total number of classes

    Returns:
        float: Mean Average Precision across all classes
    """

    average_precisions = []
    epsilon = 1e-6  # for numerical stability

    # Loop through each class
    for c in range(num_classes):

        detections = []
        ground_truths = []

        # Collect only boxes that belong to current class
        for pred in pred_boxes:
            if pred[1] == c:
                detections.append(pred)

        for gt in true_boxes:
            if gt[1] == c:
                ground_truths.append(gt)

        # Count how many ground-truth boxes exist per image
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # Convert counts into tensors to track matched boxes
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # Sort detections by confidence score (descending)
        detections.sort(key=lambda x: x[2], reverse=True)

        TP = torch.zeros(len(detections))
        FP = torch.zeros(len(detections))
        total_true_bboxes = len(ground_truths)

        # Skip class if no ground-truth exists
        if total_true_bboxes == 0:
            continue

        # Check each detection
        for detection_idx, detection in enumerate(detections):

            # Get ground-truth boxes from the same image
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            best_iou = 0
            best_gt_idx = 0

            # Find the best matching ground-truth box
            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            # Determine if detection is TP or FP
            if best_iou > iou_threshold:
                # Check if this ground-truth box was already detected
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1
            else:
                FP[detection_idx] = 1

        # Compute Precision and Recall
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)

        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)

        # Add starting points for numerical integration
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))

        # Area under Precision-Recall curve
        average_precisions.append(torch.trapz(precisions, recalls))

    # Final mean across classes
    return sum(average_precisions) / len(average_precisions)

print(f'Ok!')