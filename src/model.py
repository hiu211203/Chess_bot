import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=1,
        is_pooling=False,
        is_normalization=True,
        activation=nn.ReLU(inplace=True),
    ):
        super(ConvBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.MaxPool2d(2, 2) if is_pooling else nn.Identity(),
            nn.BatchNorm2d(out_channels) if is_normalization else nn.Identity(),
            activation,
        )

    def forward(self, x):
        return self.conv(x)


class ChessNet(nn.Module):
    def __init__(
        self,
        num_channels=6,  # number of channels in input image
        num_classes=64,  # prob distribution over 8x8 board
        activation=nn.ReLU(),
        dropout=0.6,
    ):
        super(ChessNet, self).__init__()

        self.conv1 = ConvBlock(
            in_channels=num_channels,
            out_channels=32,
            kernel_size=3,
            activation=activation,
        )
        self.conv2 = ConvBlock(
            in_channels=32,
            out_channels=128,
            kernel_size=3,
            activation=activation,
        )
        self.conv3 = ConvBlock(
            in_channels=128,
            out_channels=256,
            kernel_size=3,
            activation=activation,
        )
        self.conv4 = ConvBlock(
            in_channels=256,
            out_channels=128,
            kernel_size=3,
            activation=nn.Identity(),
        )
        self.conv4_activation = nn.LeakyReLU(inplace=True)
        self.conv5 = ConvBlock(
            in_channels=128, out_channels=64, kernel_size=3, activation=activation
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4096, 256),
            activation,
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            activation,
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x_cloned_conv2 = x.clone()
        x = self.conv3(x)
        x = self.conv4(x)
        # create skip connection from output of conv2 to input of conv5
        x += x_cloned_conv2
        # activation function is applied on output of conv4
        x = self.conv4_activation(x)
        x = self.conv5(x)
        return self.fc(x)

    def predict_proba(self, x):
        return torch.softmax(self.forward(x), dim=-1)
