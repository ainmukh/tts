import torch.nn as nn
from ..sublayers import FFTBlock, PositionalEncoder


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
        # self.pos_encoder = PositionalEncoder()
        self.layers = nn.Sequential(*[
            FFTBlock(
                hidden_size, hidden_size, attn_heads, cnn_out_channels, kernel_size, p, groups
            ) for _ in range(n_layers)
        ])

    def forward(self, batch):
        # x = batch.hiddens
        # x = self.pos_encoder(x)
        batch = self.layers(batch)
        # for i, layer in enumerate(self.layers):
        #     x, attn = layer(x)
        #     if i == 0:
        #         batch.__setattr__(f'attn', attn)

        batch.melspec_pred = batch.hiddens
        return batch
