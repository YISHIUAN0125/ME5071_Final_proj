# Domain Discriminator

import torch
import torch.nn as nn
import torchvision
from torch.autograd import Function

# gradiant revert
class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class Discriminator(nn.Module):
    def __init__(self, in_channels=256): # ResNet FPN output layer 256
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1), # 1x1
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(64, 1) # Logit (Source vs Target)
        )

    def forward(self, x, alpha=1.0):
        x = GradientReversal.apply(x, alpha)
        return self.net(x)