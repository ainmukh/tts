import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 384, hidden_size: int = 384, heads: int = 2):
        super(MultiHeadAttention, self).__init__()
        self.heads = heads
        self.hidden_size = hidden_size
        self.scaler = np.sqrt(hidden_size)

        self.q = nn.Linear(emb_size, heads * hidden_size)
        self.k = nn.Linear(emb_size, heads * hidden_size)
        self.v = nn.Linear(emb_size, heads * hidden_size)

        self.output = nn.Linear(heads * hidden_size, emb_size)

    def project(self, x, projector, batch_size):
        return projector(x)\
            .reshape(batch_size, -1, self.heads, self.hidden_size)\
            .permute(0, 2, 1, 3)\
            .reshape(batch_size * self.heads, -1, self.hidden_size)

    def forward(self, x, mask):
        batch_size = x.size(0)
        q = self.project(x, self.q, batch_size)
        k = self.project(x, self.k, batch_size)
        v = self.project(x, self.v, batch_size)

        scaled_product = torch.bmm(q, k.transpose(2, 1)) / self.scaler
        scaled_product.masked_fill_(mask.unsqueeze(1).unsqueeze(2), -10 ** 9)

        scores = F.softmax(scaled_product, -1)
        attention = torch.bmm(scores, v)

        attention = attention\
            .reshape(batch_size, self.heads, -1, self.hidden_size)\
            .permute(0, 2, 1, 3)\
            .reshape(batch_size, -1, self.heads * self.hidden_size)
        output = self.output(attention)
        return output
