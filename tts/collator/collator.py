import torch
from dataclasses import dataclass
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple, Dict, Optional
from .utils import MelSpectrogram, MelSpectrogramConfig
from ..aligner import GraphemeAligner


@dataclass
class Batch:
    waveform: torch.Tensor
    waveform_length: torch.Tensor
    melspec: torch.Tensor
    durations: torch.Tensor
    transcript: List[str]
    tokens: torch.Tensor
    token_lengths: torch.Tensor
    melspec_pred: Optional[torch.Tensor] = None
    durations_pred: Optional[torch.Tensor] = None

    def to(self, device: torch.device) -> 'Batch':
        for key, value in self.__dict__.items():
            if torch.is_tensor(value):
                value = value.to(device)
                self.__setattr__(key, value)
        return self

    def __getitem__(self, key):
        return self.__dict__[key]


class LJSpeechCollator:
    def __init__(self):
        self.melspec = MelSpectrogram(MelSpectrogramConfig())
        self.aligner = GraphemeAligner()

    def __call__(self, instances: List[Tuple]) -> Batch:
        waveform, waveforn_length, transcript, tokens, token_lengths = list(
            zip(*instances)
        )

        waveform = pad_sequence([
            waveform_[0] for waveform_ in waveform
        ]).transpose(0, 1)
        waveform_length = torch.cat(waveforn_length)

        melspec = self.melspec(waveform)

        durations = self.aligner(
            waveform, waveform_length, transcript
        )

        tokens = pad_sequence([
            tokens_[0] for tokens_ in tokens
        ]).transpose(0, 1)
        token_lengths = torch.cat(token_lengths)

        return Batch(
            waveform, waveform_length, melspec, durations,
            transcript, tokens, token_lengths
        )
