import torch
import torch.nn as nn
import torch.nn.functional as F


class DurationPredictor(nn.Module):
    def __init__(self, groups, hidden_size: int = 384, kernel_size: int = 3, p: float = 0.1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(
            hidden_size, hidden_size, kernel_size=kernel_size, padding=padding, groups=groups
        )
        self.ln1 = nn.LayerNorm(hidden_size)

        self.conv2 = nn.Conv1d(
            hidden_size, hidden_size, kernel_size=kernel_size, padding=padding, groups=groups
        )
        self.ln2 = nn.LayerNorm(hidden_size)

        self.dropout = nn.Dropout(p)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x.transpose(-1, -2))).transpose(-1, -2)
        x = self.ln1(x)

        x = F.relu(self.conv2(x.transpose(-1, -2))).transpose(-1, -2)
        x = self.ln2(x)

        x = self.linear(x)
        return x
