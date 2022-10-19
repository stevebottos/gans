import torch.nn as nn

from models.dcgan import Discriminator
from cfg import CFG


class UpBlock(nn.Module):
    # Upsample layer inspiration taken from https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
    # with some modifications
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Generator(nn.Module):
    def __init__(self, z_dim, conv_dim, channels):
        super(Generator, self).__init__()

        self.conv1 = UpBlock(z_dim, conv_dim * 8)
        self.conv2 = UpBlock(conv_dim * 8, conv_dim * 7)
        self.conv3 = UpBlock(conv_dim * 7, conv_dim * 6)
        self.conv4 = UpBlock(conv_dim * 6, conv_dim * 5)
        self.conv5 = UpBlock(conv_dim * 5, conv_dim * 4)
        self.conv6 = UpBlock(conv_dim * 4, conv_dim * 3)
        self.final = nn.Sequential(
            nn.Conv2d(conv_dim * 3, channels, kernel_size=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        z = z.view(z.size(0), -1, 1, 1)  # (*, 100, 1, 1)
        z = self.conv1(z)
        z = self.conv2(z)
        z = self.conv3(z)
        z = self.conv4(z)
        z = self.conv5(z)
        z = self.conv6(z)
        z = self.final(z)
        return z


def get_models():
    assert CFG.imsize == 64, "imsize must be 64"
    G = Generator(
        z_dim=CFG.z_dim,
        conv_dim=CFG.g_conv_dim,
        channels=CFG.channels,
    ).to(CFG.device)

    D = Discriminator(
        image_size=CFG.imsize,
        conv_dim=CFG.d_conv_dim,
        channels=CFG.channels,
    ).to(CFG.device)

    return G, D
