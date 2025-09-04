import torch
from torch import nn


class TokenEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int):
        super().__init__()
        self._embeddings = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=emb_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get corresponding embeddings to tokens
        return self._embeddings(x)


class PositionalEmbeddings(nn.Module):
    def __init__(self, max_seq_len: int, emb_size: int):
        super().__init__()
        self._embeddings = nn.Embedding(
            num_embeddings=max_seq_len, embedding_dim=emb_size
        )

    def forward(self, seq_len: int) -> torch.Tensor:
        # Get corresponding positional embeddings to tokens in sequence
        return self._embeddings.weight[:seq_len, :]
