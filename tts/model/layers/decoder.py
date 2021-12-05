import torch
import torch.nn as nn
from ..sublayers import FFTBlock


class Decoder(nn.Module):
    def __init__(self,
                 n_layers,
                 hidden_size, attn_heads,
                 cnn_out_channels, kernel_size, p, groups):
        super(Decoder, self).__init__()

        # self.layers = nn.ModuleList([
        #     FFTBlock(
        #         hidden_size, hidden_size, attn_heads, cnn_out_channels, kernel_size, p, groups
        #     ) for _ in range(n_layers)
        # ])
        self.layers = nn.Sequential(*[
            FFTBlock(
                hidden_size, hidden_size, attn_heads, cnn_out_channels, kernel_size, p, groups
            ) for _ in range(n_layers)
        ])

    def forward(self, batch):
        # x = batch.phoneme
        # x = self.layers(x)
        batch.hiddens = self.embedding(batch.hiddens)

        melspec_mask = torch.zeros(batch.melspec.size(0), batch.melspec.size(1), batch.melspec_length.max())
        for i in range(melspec_mask.size(0)):
            melspec_mask[i, :, batch.melspec_length[i]:] = 1
        batch.attn_mask = melspec_mask \
            .repeat(1, self.heads) \
            .reshape(batch.hiddens.size(0) * self.heads, -1)\
            .to(batch.hiddens.device)
        batch = self.layers(batch)
        # for i, layer in enumerate(self.layers):
        #     x, attn = layer(x)
        #     if i == 0:
        #         batch.__setattr__(f'attn', attn)

        batch.melspec_pred = batch.hiddens
        return batch
