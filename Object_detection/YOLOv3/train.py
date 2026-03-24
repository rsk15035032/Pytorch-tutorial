"""
Train YOLOv3 on Pascal VOC / COCO

Optimized for:
- Low-RAM laptops
- Faster GPU training
- Stable training loop
- Clean interview-ready code
"""

import config
import torch
import torch.optim as optim
from tqdm import tqdm

from model import YOLOv3
from loss import YoloLoss

from utils import (
    mean_average_precision,
    get_evaluation_bboxes,
    save_checkpoint,
    load_checkpoint,
    check_class_accuracy,
    get_loaders,
)

import warnings
warnings.filterwarnings("ignore")


# Faster training on GPU
torch.backends.cudnn.benchmark = True


# =========================================================
# Training Function
# =========================================================
def train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors):
    loop = tqdm(train_loader, leave=True)
    losses = []

    model.train()

    for batch_idx, (x, y) in enumerate(loop):

        x = x.to(config.DEVICE)
        y0, y1, y2 = (
            y[0].to(config.DEVICE),
            y[1].to(config.DEVICE),
            y[2].to(config.DEVICE),
        )

        # Mixed Precision Training (only useful for GPU)
        with torch.cuda.amp.autocast(enabled=config.DEVICE == "cuda"):
            outputs = model(x)

            loss = (
                loss_fn(outputs[0], y0, scaled_anchors[0])
                + loss_fn(outputs[1], y1, scaled_anchors[1])
                + loss_fn(outputs[2], y2, scaled_anchors[2])
            )

        losses.append(loss.item())

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        mean_loss = sum(losses) / len(losses)
        loop.set_postfix(loss=mean_loss)

    return mean_loss


# =========================================================
# Main Training Pipeline
# =========================================================
def main():

    print("Using device:", config.DEVICE)

    # Model
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    # Loss function
    loss_fn = YoloLoss()

    # Gradient scaler (for mixed precision)
    scaler = torch.cuda.amp.GradScaler(enabled=config.DEVICE == "cuda")

    # Load dataset
    train_loader, test_loader, train_eval_loader = get_loaders(
        train_csv_path=config.DATASET + "/train.csv",
        test_csv_path=config.DATASET + "/test.csv",
    )

    # Load checkpoint if needed
    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_FILE,
            model,
            optimizer,
            config.LEARNING_RATE,
        )

    # Scale anchors according to grid size
    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)

    # =====================================================
    # Training Loop
    # =====================================================
    for epoch in range(config.NUM_EPOCHS):

        print(f"\nEpoch [{epoch+1}/{config.NUM_EPOCHS}]")

        loss = train_fn(
            train_loader,
            model,
            optimizer,
            loss_fn,
            scaler,
            scaled_anchors,
        )

        print(f"Mean Training Loss: {loss:.4f}")

        # Save model every 5 epochs
        if config.SAVE_MODEL and epoch % 5 == 0:
            save_checkpoint(model, optimizer, filename=config.CHECKPOINT_FILE)

        # Evaluate model every 3 epochs
        if epoch > 0 and epoch % 3 == 0:

            print("\nChecking accuracy...")
            check_class_accuracy(model, test_loader, threshold=config.CONF_THRESHOLD)

            print("\nCalculating mAP...")

            pred_boxes, true_boxes = get_evaluation_bboxes(
                test_loader,
                model,
                iou_threshold=config.NMS_IOU_THRESH,
                anchors=config.ANCHORS,
                threshold=config.CONF_THRESHOLD,
            )

            mapval = mean_average_precision(
                pred_boxes,
                true_boxes,
                iou_threshold=config.MAP_IOU_THRESH,
                box_format="midpoint",
                num_classes=config.NUM_CLASSES,
            )

            print(f"mAP: {mapval.item():.4f}")

            model.train()


# =========================================================
# Run Training
# =========================================================
if __name__ == "__main__":
    main()