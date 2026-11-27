from pathlib import Path

import numpy as np

from language_model.tokenizer import Tokenizer

DATA = Path("data")
BPE = Path("artifacts/tinystories_bpe_10k")
END = "<|endoftext|>"


def encode_sample(tokenizer, source, destination, story_limit):
    token_ids = []
    story_lines = []
    story_count = 0

    with source.open(encoding="utf-8") as file:
        for line in file:
            if line.strip() == END:
                story = "".join(story_lines) + END
                token_ids.extend(tokenizer.encode(story))
                story_lines.clear()
                story_count += 1
                if story_count == story_limit:
                    break
            else:
                story_lines.append(line)

        if story_count != story_limit:
            raise ValueError(f"Found only {story_count} complete stories in {source}")

        tokens = np.asarray(token_ids, dtype=np.uint16)
        np.save(destination, tokens)
        print(f"{destination}: {story_count:,} stories, {tokens.size:,} tokens")


if __name__ == "__main__":
    tokenizer = Tokenizer.from_files(str(BPE / "vocab.pkl"), str(BPE / "merges.pkl"), special_tokens=[END])

    encode_sample(tokenizer, DATA / "TinyStoriesV2-GPT4-train.txt", DATA / "tiny-train.npy", story_limit=1000)
    encode_sample(tokenizer, DATA / "TinyStoriesV2-GPT4-valid.txt", DATA / "tiny-valid.npy", story_limit=100)
