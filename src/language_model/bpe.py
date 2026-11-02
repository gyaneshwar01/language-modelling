import os
from collections import Counter, defaultdict
from multiprocessing import Pool
from typing import BinaryIO

import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def find_chunk_boundaries(file: BinaryIO, desired_num_chunks: int, split_special_tokens: bytes) -> list[int]:
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead 4KB at a time

    for bi in range(1, len(chunk_boundaries) - 1):  # all except first and last boundary
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)

            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            found_at = mini_chunk.find(split_special_tokens)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # may return fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pretokenize_chunk(input_path: str | os.PathLike, start: int, end: int, special_tokens: list[str]) -> Counter[str]:
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    pattern = "|".join(re.escape(t) for t in special_tokens)
    docs = re.split(pattern, chunk)

    counts: Counter[str] = Counter()
    for doc in docs:
        counts.update(m.group() for m in re.finditer(PAT, doc))

    return counts


def train_bpe(
    input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab: dict[int, bytes] = {}

    for i, token in enumerate(special_tokens):
        vocab[i] = token.encode("utf-8")

    for i in range(256):
        vocab[len(vocab)] = bytes([i])

    num_processes = os.cpu_count() or 1

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes * 4, b"<|endoftext|>")

    jobs = [(input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]

    with Pool(num_processes) as pool:
        results = pool.starmap(pretokenize_chunk, jobs)

    str_counts: Counter[str] = Counter()
    for counts in results:
        str_counts.update(counts)

    words: list[tuple[bytes, ...]] = []  # words[i] = list of bytes tokens
    freqs: list[int] = []  # freqs[i] = count of that word
    for word, freq in str_counts.items():
        words.append(tuple(bytes([x]) for x in word.encode()))
        freqs.append(freq)

    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    pair_to_words: dict[tuple[bytes, bytes], set[int]] = defaultdict(set)

    for i in range(len(words)):
        word = words[i]
        for pair in zip(word, word[1:]):
            pair_counts[pair] += freqs[i]
            pair_to_words[pair].add(i)

    merges: list[tuple[bytes, bytes]] = []

    num_merges = vocab_size - len(vocab)

    for _ in range(num_merges):
        if not pair_counts:
            break
        best = max(pair_counts, key=lambda p: (pair_counts[p], p))
        merges.append(best)
        vocab[len(vocab)] = best[0] + best[1]

        affected = pair_to_words[best].copy()
        del pair_to_words[best]

        for idx in affected:
            word = words[idx]
            for pair in zip(word, word[1:]):
                pair_counts[pair] -= freqs[idx]
                if pair_counts[pair] == 0:
                    del pair_counts[pair]

            new_word = apply_merge(word, best)
            words[idx] = new_word
            for pair in zip(new_word, new_word[1:]):
                pair_counts[pair] += freqs[idx]
                pair_to_words[pair].add(idx)

    return vocab, merges


def apply_merge(word: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    new_word = []
    a, b = pair

    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
            new_word.append(a + b)
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    return tuple(new_word)
