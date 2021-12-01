import torch
import torch.nn as nn
from ..sublayers import DurationPredictor


class LengthRegulator(nn.Module):
    def __init__(self, emb_size: int, groups: int):
        super().__init__()
        self.duration_predictor = DurationPredictor(groups)
        self.silence_token = torch.Tensor([-11.5129251] * emb_size)

    def forward(self, phoneme, durations=None):
        # add silence to the end of the phoneme
        # phoneme = batch.phoneme
        silence = self.silence_token.broadcast_to(phoneme.size(0), 1, -1)
        phoneme = torch.cat((phoneme, silence), 1)

        durations_pred = self.duration_predictor(phoneme)
        if not self.training:
            durations = durations_pred

        # if self.training:
        #     durations = self.aligner(batch.waveform, batch.waveform_length, batch.transcript)
        #     # scale by waveform domain
        #     durations = (durations * batch.waveform_length.reshape(-1, 1)).int()
        #     silence_duration = (batch.waveform_length - durations.sum(-1)).reshape(-1, 1)
        #     durations = torch.cat((durations, silence_duration), 1)

        phoneme = regulate(phoneme, durations)
        return phoneme, durations_pred


@torch.no_grad()
def regulate(phoneme, durations):
    mask = torch.zeros(phoneme.size(0), phoneme.size(1), durations.sum(-1).max)
    for i in range(mask.shape[0]):
        cur_length = 0
        for j in range(mask.shape[1]):
            mask[i, j, cur_length: cur_length + durations[i][j]] = 1
            cur_length += durations[i][j]

    phoneme = phoneme.transpose(-1, -2) @ mask
    phoneme = phoneme.transpose(-1, -2)
    return phoneme