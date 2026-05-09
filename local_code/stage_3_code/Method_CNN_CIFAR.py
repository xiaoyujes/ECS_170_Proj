from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Convolution + BatchNorm + ReLU block."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Method_CNN_CIFAR(nn.Module):
    """
    CNN classifier for CIFAR-10.

    Supported variants:
    - "baseline": standard 3-stage CNN
    - "dropout": same as baseline with stronger dropout in classifier
    - "deep": deeper CNN with one additional convolution per stage
    - "kernel5": uses 5x5 kernels in the first two convolution blocks
    """

    def __init__(self, num_classes: int = 10, variant: str = "baseline"):
        super().__init__()
        self.num_classes = num_classes
        self.variant = variant

        if variant not in {"baseline", "dropout", "deep", "kernel5"}:
            raise ValueError("variant must be one of: baseline, dropout, deep, kernel5")

        dropout_rate = 0.50 if variant == "dropout" else 0.25

        if variant == "kernel5":
            self.features = nn.Sequential(
                ConvBlock(3, 32, kernel_size=5, padding=2),
                nn.MaxPool2d(kernel_size=2, stride=2),  # 32 -> 16
                ConvBlock(32, 64, kernel_size=5, padding=2),
                nn.MaxPool2d(kernel_size=2, stride=2),  # 16 -> 8
                ConvBlock(64, 128, kernel_size=3, padding=1),
                nn.MaxPool2d(kernel_size=2, stride=2),  # 8 -> 4
            )
        elif variant == "deep":
            self.features = nn.Sequential(
                ConvBlock(3, 32),
                ConvBlock(32, 32),
                nn.MaxPool2d(kernel_size=2, stride=2),  # 32 -> 16
                ConvBlock(32, 64),
                ConvBlock(64, 64),
                nn.MaxPool2d(kernel_size=2, stride=2),  # 16 -> 8
                ConvBlock(64, 128),
                ConvBlock(128, 128),
                nn.MaxPool2d(kernel_size=2, stride=2),  # 8 -> 4
            )
        else:
            self.features = nn.Sequential(
                ConvBlock(3, 32),
                nn.MaxPool2d(kernel_size=2, stride=2),  # 32 -> 16
                ConvBlock(32, 64),
                nn.MaxPool2d(kernel_size=2, stride=2),  # 16 -> 8
                ConvBlock(64, 128),
                nn.MaxPool2d(kernel_size=2, stride=2),  # 8 -> 4
            )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        logits = self.classifier(x)
        return logits


if __name__ == "__main__":
    model = Method_CNN_CIFAR(variant="baseline")
    sample = torch.randn(4, 3, 32, 32)
    output = model(sample)
    print(model)
    print("Output shape:", output.shape)  # should be [4, 10]
