import torch
from torch import Tensor
from torch.nn import MSELoss, L1Loss
from typing import Tuple


class MSELossWrapper(MSELoss):
    def forward(self, batch, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        melspec_mask = torch.ones(batch.melspec.size(0), batch.melspec.size(1), batch.melspec_length.max())
        for i in range(melspec_mask.size(0)):
            melspec_mask[i, :, batch.melspec_length[i]:] = 0
        melspec_mask = melspec_mask.to(batch.melspec.device)

        melspec = batch.melspec * melspec_mask
        melspec_pred = batch.melspec_pred * melspec_mask
        # melspec, melspec_pred = batch.melspec, batch.melspec_pred
        melspec_loss = L1Loss()(melspec_pred, melspec)

        durations, durations_pred = torch.log1p_(batch.durations), batch.durations_pred
        length_loss = super().forward(durations_pred, durations)

        return melspec_loss, length_loss
