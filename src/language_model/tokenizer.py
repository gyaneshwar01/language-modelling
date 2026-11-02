import pickle
from collections.abc import Iterable, Iterator

import regex as re

from language_model.bpe import PAT, apply_merge


class Tokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        self.vocab = dict(vocab)
        self.merges = merges
        self.special_tokens = special_tokens or []

        for tok in self.special_tokens:
            tok_bytes = tok.encode("utf-8")
            if tok_bytes not in self.vocab.values():
                self.vocab[len(self.vocab)] = tok_bytes

        if self.special_tokens:
            sorted_tokens = sorted(self.special_tokens, key=len, reverse=True)
            escaped = [re.escape(t) for t in sorted_tokens]
            self.special_pattern = "(" + "|".join(escaped) + ")"
        else:
            self.special_pattern = None

        self.token_to_id = {tok: i for i, tok in self.vocab.items()}
        self.merge_rank = {pair: i for i, pair in enumerate(self.merges)}

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[x] for x in ids).decode("utf-8", errors="replace")

    def _encode_pretoken(self, pretoken: str) -> list[int]:
        parts = tuple(bytes([x]) for x in pretoken.encode("utf-8"))

        while len(parts) > 1:
            pairs = zip(parts, parts[1:])
            best = min(pairs, key=lambda p: self.merge_rank.get(p, float("inf")))
            if best not in self.merge_rank:
                break
            parts = apply_merge(parts, best)

        ids = []
        for part in parts:
            ids.append(self.token_to_id[part])

        return ids

    def encode(self, text: str) -> list[int]:
        ids = []
        if self.special_pattern:
            pieces = re.split(self.special_pattern, text)
        else:
            pieces = [text]

        for p in pieces:
            if p == "":
                continue
            elif p in self.special_tokens:
                ids.append(self.token_to_id[p.encode("utf-8")])
            else:
                for m in re.finditer(PAT, p):
                    ids.extend(self._encode_pretoken(m.group()))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            yield from self.encode(chunk)


if __name__ == "__main__":
    vocab = {0: b" ", 1: b"a", 2: b"c", 3: b"e", 4: b"h", 5: b"t", 6: b"th", 7: b" c", 8: b" a", 9: b"the", 10: b" at"}
    merges = [(b"t", b"h"), (b" ", b"c"), (b" ", b"a"), (b"th", b"e"), (b" a", b"t")]

    # 1. The handout's full example
    t = Tokenizer(vocab, merges)
    print(t.encode("the cat ate"))  # expect: [9, 7, 1, 5, 10, 3]

    # 2. With a special token (it gets ID 11)
    t = Tokenizer(vocab, merges, ["<|endoftext|>"])
    ids = t.encode("the cat<|endoftext|> ate")
    print(ids)  # expect: [9, 7, 1, 5, 11, 10, 3]
    print(t.decode(ids))  # expect: the cat<|endoftext|> ate

    # 3. Special token at the very start (produces an empty piece)
    print(t.encode("<|endoftext|>the"))  # expect: [11, 9]

    # 4. Overlapping special tokens (IDs 11 and 12)
    t = Tokenizer(vocab, merges, ["<|endoftext|>", "<|endoftext|><|endoftext|>"])
    print(t.encode("<|endoftext|><|endoftext|>"))  # expect: [12]   (not [11, 11])

    # 5. No special tokens, empty text
    t = Tokenizer(vocab, merges)
    print(t.encode(""))  # expect: []

    t = Tokenizer(vocab, merges, ["<|endoftext|>"])
    chunks = ["the cat", "<|endoftext|>", " ate"]
    print(list(t.encode_iterable(chunks)))  # expect: [9, 7, 1, 5, 11, 10, 3]
