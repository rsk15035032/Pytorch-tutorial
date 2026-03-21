import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


# =============================================================
# Double Convolution Block
# =============================================================

class DoubleConv(nn.Module):
    """
    Core building block used in UNet.

    Structure:
    Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(DoubleConv, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# =============================================================
# UNet Architecture (Clean + Interview Ready + GPU Ready)
# =============================================================

class UNET(nn.Module):
    """
    Implementation of U-Net for image segmentation.

    Features:
    - Works on CPU and GPU
    - Supports any image size
    - Clean encoder–decoder structure
    - Easy to explain in interviews
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features=None,
    ):
        super(UNET, self).__init__()

        # Default feature sizes if not provided
        if features is None:
            features = [64, 128, 256, 512]

        self.downs = nn.ModuleList()   # Encoder blocks
        self.ups = nn.ModuleList()     # Decoder blocks
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # =========================================================
        # Encoder (Downsampling path)
        # =========================================================
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # =========================================================
        # Bottleneck (deepest layer of UNet)
        # =========================================================
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # =========================================================
        # Decoder (Upsampling path)
        # =========================================================
        for feature in reversed(features):

            # Upsample using ConvTranspose2d
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2,
                    feature,
                    kernel_size=2,
                    stride=2,
                )
            )

            # After concatenation apply DoubleConv
            self.ups.append(DoubleConv(feature * 2, feature))

        # Final 1x1 convolution to map features -> segmentation mask
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    # =============================================================
    # Forward pass
    # =============================================================
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        skip_connections = []

        # ---------------- Encoder ----------------
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # ---------------- Bottleneck ----------------
        x = self.bottleneck(x)

        # Reverse skip connections for decoder
        skip_connections = skip_connections[::-1]

        # ---------------- Decoder ----------------
        for idx in range(0, len(self.ups), 2):

            # Upsampling step
            x = self.ups[idx](x)

            # Get corresponding skip connection
            skip_connection = skip_connections[idx // 2]

            # If image size is odd, shapes may not match -> resize
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            # Concatenate skip connection with upsampled output
            x = torch.cat((skip_connection, x), dim=1)

            # Apply DoubleConv
            x = self.ups[idx + 1](x)

        return self.final_conv(x)


# =============================================================
# Simple test (CPU + GPU compatible)
# =============================================================

def test():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Random tensor (batch_size=2, RGB image 161x161)
    x = torch.randn((2, 3, 161, 161)).to(device)

    model = UNET(in_channels=3, out_channels=1).to(device)

    preds = model(x)

    print("Input shape :", x.shape)
    print("Output shape:", preds.shape)

    # Output should match input spatial size
    assert preds.shape == (2, 1, 161, 161)

    print("UNet model is working correctly! 🚀")


if __name__ == "__main__":
    test()
