import torch
import torch.nn as nn
from torchvision.models import vgg19
import config


# ==========================================================
# PERCEPTUAL LOSS (VGG19)
# Used in SRGAN for perceptual similarity
# ==========================================================

class VGGLoss(nn.Module):

    def __init__(self):
        super().__init__()

        # Load pretrained VGG19
        self.vgg = vgg19(pretrained=True).features[:35]

        # Move model to device
        self.vgg = self.vgg.to(config.DEVICE)

        # Evaluation mode (important)
        self.vgg.eval()

        # Freeze VGG parameters
        for param in self.vgg.parameters():
            param.requires_grad = False

        # Loss function
        self.loss = nn.MSELoss()

    def forward(self, input, target):

        # Extract VGG features
        vgg_input_features = self.vgg(input)
        vgg_target_features = self.vgg(target)

        # Perceptual loss
        return self.loss(vgg_input_features, vgg_target_features)