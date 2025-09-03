from collections import defaultdict
from copy import deepcopy
from typing import List
import dill


class BPE:
    def __init__(self, vocab_size: int = 50_000):
        self._vocab_size = vocab_size
        self._unique_tokens = []
        self.id2token = self.token2id = None

    def fit(self, text: str) -> None:
        # Find unique symbols & sort them
        self._unique_tokens = sorted(list(set(text)))
        assert len(self._unique_tokens) < self._vocab_size

        # Choosing the most frequent pair of tokens
        curr_tokens = list(text)
        while len(self._unique_tokens) < self._vocab_size:
            # Calculate times that token pair occurs (saving the max one)
            pair_counter = defaultdict(int)
            for i in range(len(curr_tokens) - 1):
                curr_pair = (curr_tokens[i], curr_tokens[i + 1])
                pair_counter[curr_pair] += 1
            max_pair = None
            for pair in pair_counter.keys():
                if max_pair is None or pair_counter[max_pair] < pair_counter[pair]:
                    max_pair = deepcopy(pair)

            # Adding it as new token
            self._unique_tokens.append(max_pair[0] + max_pair[1])

            # Updating tokens
            token_idx = 0
            merged_tokens = []
            while token_idx < len(curr_tokens):
                if (
                    token_idx < len(curr_tokens) - 1
                    and (curr_tokens[token_idx] == max_pair[0])
                    and (curr_tokens[token_idx + 1] == max_pair[1])
                ):
                    merged_tokens.append(max_pair[0] + max_pair[1])
                    token_idx += 2
                    continue

                merged_tokens.append(curr_tokens[token_idx])
                token_idx += 1
            curr_tokens = merged_tokens

        # Creating token ids
        self.id2token = {i: token for i, token in enumerate(self._unique_tokens)}
        self.token2id = {token: i for i, token in enumerate(self._unique_tokens)}

    def encode(self, text: str) -> List[int]:
        # Divide into simple tokens
        curr_tokens = list(text)

        # Constructing encoding
        encoding = []
        token_idx = 0
        while token_idx < len(curr_tokens):
            # Choosing best fitting token to insert
            insert_tok = curr_tokens[token_idx]
            insert_len = 1
            for tok in self.id2token.values():
                if (
                    len(tok) > insert_len
                    and "".join(curr_tokens[token_idx : token_idx + len(tok)]) == tok
                ):
                    insert_len = len(tok)
                    insert_tok = tok

            # Appending encoding
            encoding.append(self.token2id[insert_tok])
            token_idx += insert_len

        return encoding

    def decode(self, enc: List[int]) -> str:
        # Decoding string by token ids
        return "".join([self.id2token[tok] for tok in enc])

    def save(self, filename):
        with open(filename, "wb") as f:
            dill.dump(self, f)

    @classmethod
    def load(cls, filename):
        with open(filename, "rb") as f:
            obj = dill.load(f)
        return obj
