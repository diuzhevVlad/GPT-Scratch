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

    def generate(
        self,
        x: torch.Tensor,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            part = x[:, -self._max_seq_len :]
            logits = self.forward(part)[:, -1, :] / temperature
            prob = nn.functional.softmax(
                logits, dim=-1
            )  # Get prediction from last logit vector
            if do_sample:
                if top_k:
                    sorted_prob, sorted_idx = torch.sort(prob, dim=1, descending=True)
                    logits[
                        torch.arange(logits.size(0)).unsqueeze(1), sorted_idx[:, top_k:]
                    ] = -float("inf")
                    prob = nn.functional.softmax(logits, dim=-1)
                if top_p:
                    sorted_prob, sorted_idx = torch.sort(prob, dim=1, descending=True)
                    cum_sorted_prob = torch.cumsum(sorted_prob, dim=1)

                    nucleus_mask = cum_sorted_prob > top_p
                    nucleus_mask[:, 0] = 0

                    remove_mask = torch.zeros_like(nucleus_mask)
                    remove_mask.scatter_(1, sorted_idx, nucleus_mask)

                    logits = logits.masked_fill(remove_mask, -float("inf"))
                    prob = nn.functional.softmax(logits, dim=-1)

                new_col = torch.multinomial(prob, 1)
            else:
                new_col = torch.argmax(prob, dim=-1, keepdim=True)
            x = torch.cat([x, new_col], dim=1)
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
