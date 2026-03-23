"""
YOLOv1 Architecture (with BatchNorm added)

Clean + optimized version:
- CPU friendly
- GPU ready
- Cleaner architecture builder
- Proper comments only where required
"""

import torch
import torch.nn as nn


# -------------------------------------------------
# Architecture Configuration
# (kernel_size, filters, stride, padding)
# "M" -> MaxPool
# [conv1, conv2, repeats]
# -------------------------------------------------
architecture_config = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]


# -------------------------------------------------
# Convolution Block = Conv + BatchNorm + LeakyReLU
# -------------------------------------------------
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))


# -------------------------------------------------
# YOLOv1 Model
# -------------------------------------------------
class Yolov1(nn.Module):
    def __init__(self, in_channels=3, split_size=7, num_boxes=2, num_classes=20):
        super().__init__()

        self.in_channels = in_channels
        self.architecture = architecture_config

        # Backbone (Darknet)
        self.darknet = self._create_conv_layers(self.architecture)

        # Fully connected detection head
        self.fcs = self._create_fcs(split_size, num_boxes, num_classes)

    def forward(self, x):
        x = self.darknet(x)
        x = torch.flatten(x, start_dim=1)
        return self.fcs(x)

    # -------------------------------------------------
    # Build convolution backbone from config
    # -------------------------------------------------
    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for layer in architecture:

            # Standard convolution layer
            if isinstance(layer, tuple):
                kernel_size, filters, stride, padding = layer

                layers.append(
                    CNNBlock(
                        in_channels,
                        filters,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                    )
                )
                in_channels = filters

            # MaxPool layer
            elif isinstance(layer, str):
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

            # Repeated convolution blocks
            elif isinstance(layer, list):
                conv1, conv2, repeats = layer

                for _ in range(repeats):

                    layers.append(
                        CNNBlock(
                            in_channels,
                            conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3],
                        )
                    )

                    layers.append(
                        CNNBlock(
                            conv1[1],
                            conv2[1],
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3],
                        )
                    )

                    in_channels = conv2[1]

        return nn.Sequential(*layers)

    # -------------------------------------------------
    # Fully Connected Detection Head
    # -------------------------------------------------
    def _create_fcs(self, S, B, C):
        """
        Original paper:
        Linear(1024*S*S → 4096)
        Linear(4096 → S*S*(C + B*5))

        Reduced size used for CPU-friendly training
        """

        return nn.Sequential(
            nn.Linear(1024 * S * S, 496),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.0),
            nn.Linear(496, S * S * (C + B * 5)),
        )