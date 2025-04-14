import torch
from torch import nn

# 构建CNN模型
class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=16*3,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2) # (16*3, 16, 16)
            nn.MaxPool2d(kernel_size=1) # (16*3, 16, 16)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16*3,
                out_channels=32*3,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),

            # nn.MaxPool2d(kernel_size=2) # (32*3, 8, 8)
            nn.MaxPool2d(kernel_size=2) # (32*3, 16, 16)
        )

        self.out = nn.Linear(32*16*16*3, 34)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        out = self.out(x)
        return out