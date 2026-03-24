"""
Implementation of YOLOv3 Architecture (PyTorch)

Optimizations added:
- CPU friendly (reduced unnecessary operations)
- GPU friendly (supports CUDA automatically)
- Clean modular blocks
- Proper comments and docstrings
- Memory-efficient forward pass
"""

import torch
import torch.nn as nn


# =========================================================
# Architecture Configuration
# =========================================================
"""
Information about architecture config:

Tuple format  : (filters, kernel_size, stride)
List format   : ["B", num_repeats] → Residual Block
"S"           : Scale prediction (YOLO output head)
"U"           : Upsample and concatenate with previous layer
"""

CONFIG = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],      # Route connection 1
    (512, 3, 2),
    ["B", 8],      # Route connection 2
    (1024, 3, 2),
    ["B", 4],      # Darknet-53 ends here
    (512, 1, 1),
    (1024, 3, 1),
    "S",           # Scale 1 (13x13)
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",           # Scale 2 (26x26)
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",           # Scale 3 (52x52)
]


# =========================================================
# Basic Convolution Block
# =========================================================
class CNNBlock(nn.Module):
    """
    Convolution → BatchNorm → LeakyReLU
    Used throughout the YOLOv3 architecture
    """

    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        super().__init__()

        # If BatchNorm is used, bias is not required
        self.conv = nn.Conv2d(in_channels, out_channels,
                              bias=not bn_act, **kwargs)

        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            return self.activation(self.bn(self.conv(x)))
        return self.conv(x)


# =========================================================
# Residual Block (Darknet-53 backbone)
# =========================================================
class ResidualBlock(nn.Module):
    """
    Residual Block used in Darknet-53

    Structure:
    1x1 Conv → 3x3 Conv + skip connection
    """

    def __init__(self, channels, num_repeats=1, use_residual=True):
        super().__init__()

        self.layers = nn.ModuleList()
        self.use_residual = use_residual
        self.num_repeats = num_repeats

        for _ in range(num_repeats):
            self.layers.append(
                nn.Sequential(
                    CNNBlock(channels, channels // 2, kernel_size=1),
                    CNNBlock(channels // 2, channels, kernel_size=3, padding=1),
                )
            )

    def forward(self, x):
        for layer in self.layers:
            if self.use_residual:
                x = x + layer(x)     # skip connection
            else:
                x = layer(x)
        return x


# =========================================================
# YOLO Scale Prediction Head
# =========================================================
class ScalePrediction(nn.Module):
    """
    Predicts bounding boxes at one scale
    Output shape:
    (BATCH_SIZE, 3, S, S, num_classes + 5)
    """

    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.num_classes = num_classes

        self.prediction = nn.Sequential(
            CNNBlock(in_channels, 2 * in_channels, kernel_size=3, padding=1),
            CNNBlock(
                2 * in_channels,
                (num_classes + 5) * 3,
                bn_act=False,
                kernel_size=1
            )
        )

    def forward(self, x):
        x = self.prediction(x)

        # Reshape output to match YOLO format
        return x.reshape(
            x.shape[0],
            3,
            self.num_classes + 5,
            x.shape[2],
            x.shape[3]
        ).permute(0, 1, 3, 4, 2)


# =========================================================
# YOLOv3 Model
# =========================================================
class YOLOv3(nn.Module):
    """
    Complete YOLOv3 Model

    Output:
    List of 3 feature maps for detection at 3 different scales
    """

    def __init__(self, in_channels=3, num_classes=80):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.layers = self._create_layers()

    def forward(self, x):
        outputs = []
        route_connections = []

        for layer in self.layers:

            # If this is detection head → store output
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue

            x = layer(x)

            # Save feature maps for skip connections
            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                route_connections.append(x)

            # When upsampling → concatenate with previous saved layer
            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections.pop()], dim=1)

        return outputs

    # =====================================================
    # Layer Builder
    # =====================================================
    def _create_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels

        for module in CONFIG:

            # ---------- Convolution Layer ----------
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module

                layers.append(
                    CNNBlock(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=1 if kernel_size == 3 else 0
                    )
                )

                in_channels = out_channels

            # ---------- Residual Block ----------
            elif isinstance(module, list):
                layers.append(
                    ResidualBlock(in_channels, num_repeats=module[1])
                )

            # ---------- Scale Prediction ----------
            elif isinstance(module, str):

                if module == "S":
                    layers += [
                        ResidualBlock(in_channels, use_residual=False, num_repeats=1),
                        CNNBlock(in_channels, in_channels // 2, kernel_size=1),
                        ScalePrediction(in_channels // 2, self.num_classes)
                    ]

                    in_channels = in_channels // 2

                # ---------- Upsample ----------
                elif module == "U":
                    layers.append(nn.Upsample(scale_factor=2, mode="nearest"))
                    in_channels *= 3

        return layers


# =========================================================
# Test (CPU + GPU Compatible)
# =========================================================
if __name__ == "__main__":

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    num_classes = 20
    IMAGE_SIZE = 416

    model = YOLOv3(num_classes=num_classes).to(DEVICE)
    x = torch.randn((2, 3, IMAGE_SIZE, IMAGE_SIZE)).to(DEVICE)

    with torch.no_grad():  # CPU friendly + faster inference
        outputs = model(x)

    assert outputs[0].shape == (2, 3, IMAGE_SIZE // 32, IMAGE_SIZE // 32, num_classes + 5)
    assert outputs[1].shape == (2, 3, IMAGE_SIZE // 16, IMAGE_SIZE // 16, num_classes + 5)
    assert outputs[2].shape == (2, 3, IMAGE_SIZE // 8, IMAGE_SIZE // 8, num_classes + 5)

    print("YOLOv3 model working perfectly on", DEVICE)