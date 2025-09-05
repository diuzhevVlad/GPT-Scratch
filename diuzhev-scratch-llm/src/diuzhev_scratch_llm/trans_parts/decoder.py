import torch
from torch import nn
from .attention import MultiHeadAttention
from .ffn import FeedForward


class Decoder(nn.Module):
    def __init__(
        self,
        num_heads: int,
        emb_size: int,
        head_size: int,
        max_seq_len: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Initializing layers
        self._multi_head_attn = MultiHeadAttention(
            num_heads, emb_size, head_size, max_seq_len, dropout
        )
        self._layer_norm1 = nn.LayerNorm(emb_size)
        self._ff = FeedForward(emb_size, dropout=dropout)
        self._layer_norm2 = nn.LayerNorm(emb_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Decoder block with residual connections
        x = self._multi_head_attn(x) + x
        x = self._layer_norm1(x)
        x = self._ff(x) + x
        x = self._layer_norm2(x)
        return x
