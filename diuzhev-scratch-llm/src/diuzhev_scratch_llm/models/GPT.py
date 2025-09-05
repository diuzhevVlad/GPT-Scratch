import torch
from torch import nn
from ..embedding import TokenEmbeddings, PositionalEmbeddings
from ..trans_parts import Decoder


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        emb_size: int,
        num_heads: int,
        head_size: int,
        num_layers: int,
        dropout: float = 0.1,
        device: str = "cuda",
    ):
        super().__init__()

        # Initializing layers
        self._token_emb = TokenEmbeddings(vocab_size, emb_size).to(device)
        self._pos_emb = PositionalEmbeddings(max_seq_len, emb_size).to(device)
        self._dropout = nn.Dropout(dropout).to(device)
        self._decoders = nn.Sequential(
            *[
                Decoder(num_heads, emb_size, head_size, max_seq_len, dropout)
                for _ in range(num_layers)
            ]
        ).to(device)
        self._linear = nn.Linear(emb_size, vocab_size).to(device)

        # Saving data
        self._max_seq_len = max_seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._token_emb(x) + self._pos_emb(len(x[0]))  # Get embedding
        x = self._dropout(x)  # Eliminate overfitting
        x = self._decoders(x)  # Main decoders part
        x = self._linear(x)  # Final logits
        return x

    def generate(self, x: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            part = x[:, -min(self._max_seq_len, len(x)) :]
            logits = self.forward(part)[:, -1, :]
            prob = nn.functional.softmax(
                logits, dim=-1
            )  # Get prediction from last logit vector
            x = torch.cat([x, torch.argmax(prob, dim=1, keepdim=True)], dim=1)
        return x

    def save(self, path):
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "vocab_size": self.vocab_size,
                "max_seq_len": self.max_seq_len,
                "emb_size": self.emb_size,
                "num_heads": self.num_heads,
                "head_size": self.head_size,
                "num_layers": self.num_layers,
            },
            path,
        )

    @classmethod
    def load(cls, path, device):
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            vocab_size=checkpoint["vocab_size"],
            max_seq_len=checkpoint["max_seq_len"],
            emb_size=checkpoint["emb_size"],
            num_heads=checkpoint["num_heads"],
            head_size=checkpoint["head_size"],
            num_layers=checkpoint["num_layers"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
