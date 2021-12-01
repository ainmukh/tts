import torch.nn as nn
from .layers import Encoder, Decoder, LengthRegulator


class FastSpeech(nn.Module):
    def __init__(self,
                 n_layers,
                 vocabulary_size, hidden_size, attn_heads,
                 cnn_out_channels, kernel_size, p,
                 n_mels, groups):
        super(FastSpeech, self).__init__()

        self.encoder = Encoder(
            n_layers, vocabulary_size, hidden_size, attn_heads, cnn_out_channels, kernel_size, p, groups
        )

        self.length_regulator = LengthRegulator(hidden_size, groups)

        self.decoder = Decoder(
            n_layers, hidden_size, attn_heads, cnn_out_channels, kernel_size, p, groups
        )

        self.linear = nn.Linear(hidden_size, n_mels)

    def forward(self, batch):
        # phoneme = batch.phoneme
        # phoneme = self.encoder(phoneme)
        batch = self.encoder(batch)

        phoneme, durations_pred = self.length_regulator(batch.phoneme, batch.durations)
        batch.durations_pred = durations_pred

        # mel_spectrogram = self.decoder(phoneme)
        batch = self.decoder(batch)
        batch.melspec_pred = self.linear(batch.melspec_pred)
        return batch
