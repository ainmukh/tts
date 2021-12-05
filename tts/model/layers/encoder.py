import torch.nn as nn
from ..sublayers import FFTBlock


class Encoder(nn.Module):
    def __init__(self,
                 n_layers,
                 vocabulary_size, hidden_size, attn_heads,
                 cnn_out_channels, kernel_size, p, groups):
        super(Encoder, self).__init__()

        self.heads = attn_heads
        self.embedding = nn.Embedding(vocabulary_size, hidden_size)
        self.layers = nn.Sequential(*[
            FFTBlock(
                hidden_size, hidden_size, attn_heads, cnn_out_channels, kernel_size, p, groups
            ) for _ in range(n_layers)
        ])

    def forward(self, batch):
        # x = self.embedding(batch.tokens)
        # x = self.layers(x)
        batch.hiddens = self.embedding(batch.tokens)
        batch.attn_mask = (batch.tokens == 0)\
            .repeat(1, self.heads)\
            .reshape(batch.tokens.size(0) * self.heads, -1).to(batch.token.device)
        batch = self.layers(batch)
        # for i, layer in enumerate(self.layers):
        #     x, attn = layer(x)
        #     # batch.__setattr__(f'encoder_attn{i}', attn)

        # batch.phoneme = x
        return batch
