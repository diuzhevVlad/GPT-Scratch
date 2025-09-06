import torch
from typing import List


class LLMDataset(torch.utils.data.Dataset):
    def __init__(self, data: List[int], seq_len: int, device: str = "cpu"):
        super().__init__()
        self._data = torch.tensor(data, dtype=torch.int16, device=device)
        self._seq_len = seq_len

    def __len__(self):
        return self._data.size(0) - self._seq_len - 1

    def __getitem__(self, index):
        return (
            self._data[index : index + self._seq_len],
            self._data[index + 1 : index + self._seq_len + 1],
        )
