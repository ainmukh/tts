import torch.nn as nn
from ..sublayers import FFTBlock, PositionalEncoder


class Encoder(nn.Module):
    def __init__(self,
                 n_layers,
                 vocabulary_size, hidden_size, attn_heads,
                 cnn_out_channels, kernel_size, p, groups):
        super(Encoder, self).__init__()

        self.embedding = nn.Embedding(vocabulary_size, hidden_size)
        self.layers = nn.Sequential(*[
            FFTBlock(
                hidden_size, hidden_size, attn_heads, cnn_out_channels, kernel_size, p, groups
            ) for _ in range(n_layers)
        ])

    def forward(self, batch):
        x = self.embedding(batch.tokens)
        x = self.layers(x)
        # for i, layer in enumerate(self.layers):
        #     x, attn = layer(x)
        #     # batch.__setattr__(f'encoder_attn{i}', attn)

        batch.phoneme = x
        return batch
