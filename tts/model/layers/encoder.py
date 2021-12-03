import torch.nn as nn
from ..sublayers import FFTBlock, PositionalEncoder


class Encoder(nn.Module):
    def __init__(self,
                 n_layers,
                 vocabulary_size, hidden_size, attn_heads,
                 cnn_out_channels, kernel_size, p, groups):
        super(Encoder, self).__init__()

        self.embedding = nn.Embedding(vocabulary_size, hidden_size)
        # self.pos_encoder = PositionalEncoder()
        self.layers = nn.Sequential(*[
            FFTBlock(
                hidden_size, hidden_size, attn_heads, cnn_out_channels, kernel_size, p, groups
            ) for _ in range(n_layers)
        ])

    def forward(self, batch):
        batch.hiddens = self.embedding(batch.tokens)
        # x = self.pos_encoder(x)
        batch = self.layers(batch)
        # for i, layer in enumerate(self.layers):
        #     x, attn = layer(x)
        #     # batch.__setattr__(f'encoder_attn{i}', attn)

        return batch
