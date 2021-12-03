import torch.nn as nn
from . import MultiHeadAttention, Conv


class FFTBlock(nn.Module):
    def __init__(self,
                 in_size: int = 384, hidden_size: int = 384, heads: int = 2,
                 out_channels: int = 1536, kernel_size: int = 3,
                 p: float = 0.1, groups: int = 1):
        super(FFTBlock, self).__init__()

        self.attention = MultiHeadAttention(in_size, hidden_size, heads)
        self.ln1 = nn.LayerNorm(in_size)
        self.conv = Conv(in_size, out_channels, kernel_size, groups)
        self.ln2 = nn.LayerNorm(in_size)
        self.dropout = nn.Dropout(p)

    def forward(self, hiddens):
        att = self.attention(hiddens)
        res = self.ln1(att + hiddens)
        res = self.dropout(res)

        conv_out = self.conv(res.transpose(-1, -2)).transpose(-1, -2)
        res = self.ln2(conv_out + res)
        res = self.dropout(res)
        return res
