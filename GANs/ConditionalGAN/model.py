import torch
import torch.nn as nn


# -----------------------------
# Helper: initialize weights
# -----------------------------
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


# -----------------------------
# Generator (Conditional)
# -----------------------------
class Generator(nn.Module):
    def __init__(self, z_dim, num_classes, img_channels, features_g, embed_size):
        super().__init__()

        self.embed = nn.Embedding(num_classes, embed_size)

        self.net = nn.Sequential(
            # Input: Z + label embedding → (z_dim + embed_size) x 1 x 1
            self._block(z_dim + embed_size, features_g * 4, 4, 1, 0),  # 4x4
            self._block(features_g * 4, features_g * 2, 4, 2, 1),      # 8x8
            self._block(features_g * 2, features_g, 4, 2, 1),          # 16x16
            nn.ConvTranspose2d(features_g, img_channels, 4, 2, 1),     # 32x32
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, noise, labels):
        embed = self.embed(labels).unsqueeze(2).unsqueeze(3)
        x = torch.cat([noise, embed], dim=1)
        return self.net(x)


# -----------------------------
# Critic (Conditional)
# -----------------------------
class Critic(nn.Module):
    def __init__(self, img_channels, num_classes, features_d, img_size):
        super().__init__()

        self.embed = nn.Embedding(num_classes, img_size * img_size)

        self.net = nn.Sequential(
            nn.Conv2d(img_channels + 1, features_d, 4, 2, 1),  # 32→16
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d * 2, 4, 2, 1),  # 16→8
            self._block(features_d * 2, features_d * 4, 4, 2, 1),  # 8→4
            nn.Conv2d(features_d * 4, 1, 4, 2, 0),  # 4→1
        )

    def _block(self, in_channels, out_channels, kernel, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel, stride, padding, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x, labels):
        embedding = self.embed(labels).view(labels.shape[0], 1, x.shape[2], x.shape[3])
        x = torch.cat([x, embedding], dim=1)
        return self.net(x).view(-1)