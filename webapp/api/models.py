"""Definition of the models used in the application."""

import typing as tp
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """DoubleConv, BN, ReLU."""

    def __init__(
        self, in_channels: int, out_channels: int, mid_channels: tp.Optional[int] = None
    ):
        """Initialize DoubleConv.

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            mid_channels (int, optional): Number of channels in the mid layer.
                Defaults to None, in which case it is set to out_channels.
        """
        super().__init__()

        # If mid_channels is not specified, set it to out_channels.
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """Run module."""
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int):
        """Initialize Down.

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
        """
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        """Run module."""
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        """Initialize Up.

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            bilinear (bool): If bilinear, use no stride to reduce the number of channels.
                Defaults to True.
        """
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """Run module."""
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)


class OutConv(nn.Module):
    """Output layer."""

    def __init__(self, in_channels: int, out_channels: int):
        """Initialize OutConv.

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """Run module."""
        return self.conv(x)


class UNet(nn.Module):
    """UNet model."""

    def __init__(self, n_channels: int, n_classes: int, bilinear: bool = False):
        """Initialize UNet model.

        Args:
            n_channels (int): Number of input channels
            n_classes (int): Number of output channels
            bilinear (bool): Whether or not to run the model in bilinear mode.
                Defaults to False.
        """
        super().__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        """Run forward."""
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        return (
            torch.sigmoid(logits[:, 0, ...]),
            torch.sigmoid(logits[:, 1, ...]),
        )
