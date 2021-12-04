import torch
from torch import Tensor
from torch.nn import MSELoss, L1Loss
from typing import Tuple


class MSELossWrapper(MSELoss):
    def forward(self, batch, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        melspec, melspec_pred = batch.melspec, batch.melspec_pred
        melspec_loss = L1Loss()(melspec_pred, melspec)

        durations, durations_pred = torch.log1p_(batch.durations), batch.durations_pred
        length_loss = super().forward(durations_pred, durations)

        return melspec_loss, length_loss
