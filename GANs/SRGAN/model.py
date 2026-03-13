import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Basic Convolution Block used in both Generator and Discriminator.
    Consists of Conv -> BatchNorm (optional) -> Activation (optional)
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        discriminator=False,
        use_activation=True,
        use_bn=True,
        **kwargs,
    ):
        super().__init__()

        self.use_activation = use_activation

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            bias=not use_bn,
            **kwargs
        )

        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()

        self.activation = (
            nn.LeakyReLU(0.2, inplace=True)
            if discriminator
            else nn.PReLU(num_parameters=out_channels)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.use_activation:
            x = self.activation(x)
        return x


class UpsampleBlock(nn.Module):
    """
    Upsampling block using PixelShuffle.
    Converts low-resolution feature maps into higher resolution.
    """

    def __init__(self, in_channels, scale_factor):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            in_channels * scale_factor**2,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.activation = nn.PReLU(num_parameters=in_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.activation(x)
        return x


class ResidualBlock(nn.Module):
    """
    Residual Block used inside the generator.
    Helps stabilize training and improves feature learning.
    """

    def __init__(self, channels):
        super().__init__()

        self.block1 = ConvBlock(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.block2 = ConvBlock(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            use_activation=False,
        )

    def forward(self, x):
        residual = self.block1(x)
        residual = self.block2(residual)
        return x + residual


class Generator(nn.Module):
    """
    SRGAN Generator Network
    Upscales low-resolution images into high-resolution outputs.
    """

    def __init__(self, in_channels=3, num_channels=64, num_blocks=16):
        super().__init__()

        self.initial = ConvBlock(
            in_channels,
            num_channels,
            kernel_size=9,
            stride=1,
            padding=4,
            use_bn=False,
        )

        self.residuals = nn.Sequential(
            *[ResidualBlock(num_channels) for _ in range(num_blocks)]
        )

        self.conv_block = ConvBlock(
            num_channels,
            num_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            use_activation=False,
        )

        self.upsamples = nn.Sequential(
            UpsampleBlock(num_channels, scale_factor=2),
            UpsampleBlock(num_channels, scale_factor=2),
        )

        self.final = nn.Conv2d(
            num_channels,
            in_channels,
            kernel_size=9,
            stride=1,
            padding=4,
        )

    def forward(self, x):
        initial = self.initial(x)

        x = self.residuals(initial)
        x = self.conv_block(x)

        x = x + initial

        x = self.upsamples(x)

        return torch.tanh(self.final(x))


class Discriminator(nn.Module):
    """
    SRGAN Discriminator Network
    Determines whether an image is real or generated.
    """

    def __init__(
        self,
        in_channels=3,
        features=[64, 64, 128, 128, 256, 256, 512, 512],
    ):
        super().__init__()

        layers = []

        for idx, feature in enumerate(features):
            layers.append(
                ConvBlock(
                    in_channels,
                    feature,
                    kernel_size=3,
                    stride=1 + idx % 2,
                    padding=1,
                    discriminator=True,
                    use_bn=False if idx == 0 else True,
                )
            )
            in_channels = feature

        self.blocks = nn.Sequential(*layers)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(512 * 6 * 6, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
        )

    def forward(self, x):
        x = self.blocks(x)
        x = self.classifier(x)
        return x


def test():
    """
    Simple test to verify generator and discriminator outputs.
    """

    low_resolution = 24  # For 4x super-resolution -> 96x96 output
    
    # use ----> with torch.cuda.amp.autocast(): ===> for CUDA.

    x = torch.randn((5, 3, low_resolution, low_resolution))


    gen = Generator()
    disc = Discriminator()

    gen_out = gen(x)
    disc_out = disc(gen_out)

    print("Generator output shape:", gen_out.shape)
    print("Discriminator output shape:", disc_out.shape)


if __name__ == "__main__":
    test()