import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights
import config

# phi_5,4 → 5th conv layer before maxpool but after activation

class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()

        # Load pretrained VGG19
        self.vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:36]

        # Move to device (CPU)
        self.vgg = self.vgg.to(config.DEVICE).eval()

        # Freeze VGG parameters
        for param in self.vgg.parameters():
            param.requires_grad = False

        self.loss = nn.MSELoss()

    def forward(self, input, target):

        # Extract features
        vgg_input_features = self.vgg(input)

        # No need to compute gradients for target
        with torch.no_grad():
            vgg_target_features = self.vgg(target)

        return self.loss(vgg_input_features, vgg_target_features)
    
