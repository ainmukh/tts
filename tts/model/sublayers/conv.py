import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
    def __init__(self,
                 in_channels: int, out_channels: int,
                 kernel_size: int, groups: int):
        super(Conv, self).__init__()

        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, groups=groups)
        self.conv2 = nn.Conv1d(out_channels, in_channels, kernel_size, padding=padding, groups=groups)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(F.relu(x))
        return x
