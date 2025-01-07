import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.tanh(self.conv1(x))
        x = self.avgpool(x)

        x = F.tanh(self.conv2(x))
        x = self.avgpool(x)

        # 按照原文的实现，这里把特征图展平后过全连接层
        # 更为合理的实现应该是过一个5*5的卷积把通道数变为120，特征图大小变为1*1，然后过全连接层
        x = x.view(-1, 16 * 5 * 5)

        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))

        return x
