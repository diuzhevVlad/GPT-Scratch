import torch
from torch import nn


class HeadAttention(nn.Module):
    def __init__(self, emb_size: int, head_size: int, max_seq_len: int):
        super().__init__()

        # Creating weight matrices
        self._Wk = nn.Linear(emb_size, head_size)
        self._Wq = nn.Linear(emb_size, head_size)
        self._Wv = nn.Linear(emb_size, head_size)
        self._mask = torch.tril(
            torch.ones((max_seq_len, max_seq_len), dtype=torch.bool)
        )

        # Saving dim info
        self._head_size = head_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get sequence len
        seq_sz = x.shape[1]

        # Main matrices
        key = self._Wk(x)
        query = self._Wq(x)
        value = self._Wv(x)

        # Attention matrix
        attention_mat = (
            torch.bmm(query, torch.transpose(key, -2, -1)) / self._head_size**0.5
        )
        attention_mat[:, ~self._mask[:seq_sz, :seq_sz]] = float("-inf")
        attention_mat = nn.functional.softmax(attention_mat, dim=2)  # Row-wise softmax

        # Att = Softmax(Q @ K^T / sqrt(head_size)) @ V
        return torch.bmm(attention_mat, value)
