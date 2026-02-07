import random
import math
import cv2
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import numpy as np
import albumentations as A


def visualize(image):
    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.imshow(image)
    plt.show()


def plot_examples(images, bboxes=None, columns=4):
    total = len(images)
    rows = math.ceil(total / columns)

    fig = plt.figure(figsize=(columns * 4, rows * 4))

    for i, img in enumerate(images, start=1):
        if bboxes is not None:
            img = visualize_bbox(img, bboxes[i - 1], class_name="shahrukh")

        ax = fig.add_subplot(rows, columns, i)
        ax.imshow(img)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


# From https://albumentations.ai/docs/examples/example_bboxes/
def visualize_bbox(img, bbox, class_name, color=(255, 0, 0), thickness=5):
    """Visualizes a single bounding box on the image"""
    x_min, y_min, x_max, y_max = map(int, bbox)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, thickness)
    return img