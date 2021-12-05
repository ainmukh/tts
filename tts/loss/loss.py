import torch
from torch import Tensor
from torch.nn import MSELoss, L1Loss
from typing import Tuple
from ..collator import Batch


class BadDurationException(Exception):
    batch: Batch


class MSELossWrapper(MSELoss):
    def forward(self, batch, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        # melspec_mask = torch.ones(batch.melspec.size(0), batch.melspec.size(1), batch.melspec_length.max())
        # for i in range(melspec_mask.size(0)):
        #     melspec_mask[i, :, batch.melspec_length[i]:] = 0
        # melspec_mask = melspec_mask.to(batch.melspec.device)
        #
        # melspec = batch.melspec * melspec_mask
        # melspec_pred = batch.melspec_pred * melspec_mask
        melspec, melspec_pred = batch.melspec, batch.melspec_pred
        melspec_loss = L1Loss()(melspec_pred, melspec)

        # durations_mask = (batch.tokens == 0).to(batch.tokens.device)
        # durations = torch.log1p_(batch.durations) * durations_mask
        # durations_pred = batch.durations_pred * durations_mask
        durations, durations_pred = torch.log1p_(batch.durations), batch.durations_pred
        try:
            length_loss = super().forward(durations_pred, durations)
        except Exception as e:
            raise BadDurationException(batch)

        return melspec_loss, length_loss
