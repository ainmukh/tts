import torch
from torch import Tensor
from torch.nn import MSELoss
from typing import Tuple


class MSELossWrapper(MSELoss):
    def forward(self, batch, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        melspec, melspec_pred = batch.melspec, batch.melspec_pred
        melspec_loss = super(MSELossWrapper, self).forward(melspec_pred, melspec)

        durations, durations_pred = batch.durations, batch.durations_pred
        length_loss = super().forward(durations_pred, torch.log1p(durations))

        return melspec_loss, length_loss
