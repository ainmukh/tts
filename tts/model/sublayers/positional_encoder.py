import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoder(nn.Module):
    def __init__(self, max_len=1000, emb_size=384):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        log = torch.log(torch.FloatTensor([10000])).item()
        div_term = torch.exp(torch.arange(0, emb_size, 2) * (-log / emb_size))
        pe = torch.zeros(1, emb_size, max_len)
        pe[0, 0::2, :] = torch.sin(position * div_term).T
        pe[0, 1::2, :] = torch.cos(position * div_term).T
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :, :x.size(2)]
