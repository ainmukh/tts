import torch
import torch.nn as nn
from ..sublayers import DurationPredictor


class LengthRegulator(nn.Module):
    def __init__(self, emb_size: int, groups: int):
        super().__init__()
        self.duration_predictor = DurationPredictor(groups)
        self.silence_token = torch.Tensor([-11.5129251] * emb_size)

    def forward(self, batch):
        # add silence to the end of the phoneme
        # phoneme = batch.phoneme
        phoneme = batch.hiddens
        durations_pred = self.duration_predictor(phoneme).squeeze(-1)
        batch.durations_pred = durations_pred

        silence = self.silence_token.broadcast_to(phoneme.size(0), 1, -1).to(device=phoneme.device)
        phoneme = torch.cat((phoneme, silence), 1)

        if not self.training:
            durations = torch.expm1(durations_pred)
        else:
            durations = batch.durations

        # if self.training:
            # durations = self.aligner(batch.waveform, batch.waveform_length, batch.transcript)
            # scale by waveform domain
        durations = (durations * batch.melspec_length.reshape(-1, 1)).int()
        silence_duration = (batch.melspec_length - durations.sum(-1)).reshape(-1, 1)
        durations = torch.cat((durations, silence_duration), 1).int()

        phoneme = regulate(phoneme, durations)

        batch.hiddens = phoneme
        return batch


@torch.no_grad()
def regulate(phoneme, durations):
    mask = torch.zeros(phoneme.size(0), phoneme.size(1), durations.sum(-1).max())
    for i in range(mask.shape[0]):
        cur_length = 0
        for j in range(mask.shape[1]):
            mask[i, j, cur_length: cur_length + durations[i][j]] = 1
            cur_length += durations[i][j]

    mask = mask.to(phoneme.device)
    phoneme = phoneme.transpose(-1, -2) @ mask
    phoneme = phoneme.transpose(-1, -2)
    return phoneme
