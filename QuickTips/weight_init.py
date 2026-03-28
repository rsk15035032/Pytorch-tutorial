"""
Simple Convolutional Neural Network (CNN) in PyTorch

Concept covered:
- Convolution layers for feature extraction
- MaxPooling for spatial downsampling
- Fully connected layer for classification
- Kaiming (He) weight initialization
"""

# ==============================
# Imports
# ==============================
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================
# CNN Model
# ==============================
class SimpleCNN(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 10):
        """
        Args:
            in_channels : number of input channels (3 for RGB, 1 for grayscale)
            num_classes : number of output classes
        """
        super().__init__()

        # ---------- Convolution Block 1 ----------
        # Input  : (N, 3, 32, 32)
        # Output : (N, 6, 32, 32)
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=3, stride=1, padding=1)

        # Reduces spatial size by half -> (32x32 → 16x16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # ---------- Convolution Block 2 ----------
        # Input  : (N, 6, 16, 16)
        # Output : (N, 16, 16, 16)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=1)

        # ---------- Fully Connected Layer ----------
        # After two poolings: 32x32 → 16x16 → 8x8
        # Final feature map = 16 channels × 8 × 8
        self.fc = nn.Linear(16 * 8 * 8, num_classes)

        # Initialize weights
        self._initialize_weights()

    # ==============================
    # Forward Pass
    # ==============================
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation logic:
        Conv -> ReLU -> Pool -> Conv -> ReLU -> Pool -> Flatten -> FC
        """

        # Feature extraction
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten (N, C, H, W) -> (N, C*H*W)
        x = torch.flatten(x, start_dim=1)

        # Classification
        x = self.fc(x)

        return x

    # ==============================
    # Weight Initialization
    # ==============================
    def _initialize_weights(self):
        """
        Kaiming initialization works best for ReLU networks.
        Helps faster convergence and prevents vanishing gradients.
        """
        for layer in self.modules():

            # Initialize Conv layers
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

            # Initialize Linear layers
            elif isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
                nn.init.constant_(layer.bias, 0)


# ==============================
# Test the Model
# ==============================
if __name__ == "__main__":

    # Create model
    model = SimpleCNN(in_channels=3, num_classes=10)

    # Print model architecture
    print(model)

    # Test with dummy input (like CIFAR-10 images)
    x = torch.randn(1, 3, 32, 32)
    y = model(x)

    print("Output shape:", y.shape)