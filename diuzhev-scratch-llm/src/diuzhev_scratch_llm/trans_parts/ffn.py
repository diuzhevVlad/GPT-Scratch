import torch
from torch import nn


class FeedForward(nn.Module):
    def __init__(self, emb_size: int, hid_size: int = -1, dropout: float = 0.1):
        """
        If **hid_size** = -1, 4 x emb_size is used
        """
        super().__init__()

        # Initializing layers
        hid_size = hid_size if hid_size != -1 else emb_size * 4
        self._linear1 = nn.Linear(emb_size, hid_size)
        self._relu = nn.ReLU()
        self._linear2 = nn.Linear(hid_size, emb_size)
        self._dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Simple FF network logic
        x = self._linear1(x)
        x = self._relu(x)
        x = self._linear2(x)
        x = self._dropout(x)
        return x
