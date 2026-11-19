from __future__ import annotations

import re
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from tqdm.auto import tqdm

import clslib


def to_vocab(name: str, tokens: list[str]) -> list[str]:
    """Apply embedding-specific token normalization."""

    if name == "numberbatch":
        return [
            re.sub(r"\d", "#", token)
            if re.search(r"\d\d", token)
            else token
            for token in tokens
        ]

    return tokens

class BiographyDataset(Dataset):
    """A dataset containing token IDs and profession labels."""

    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (
            torch.tensor(self.sequences[index], dtype=torch.long),
            torch.tensor(self.labels[index], dtype=torch.long),
        )


def collate_biographies(batch):
    """Pad sequences in a batch and return their original lengths."""

    sequences, labels = zip(*batch)

    lengths = torch.tensor(
        [len(sequence) for sequence in sequences],
        dtype=torch.long,
    )

    sequences = pad_sequence(
        sequences,
        batch_first=True,
        padding_value=0,
    )

    labels = torch.stack(labels)

    return sequences, lengths, labels


def prepare_data(
    embedding_name,
    pretrained,
    max_length=256,
    max_vocab_size=0,
):
    """
    Tokenize the Bias in Bios dataset and construct a vocabulary.

    Vocabulary is constructed exclusively from the training split.
    Index 0 = PAD, index 1 = UNK.
    """

    tokenizer = clslib.BasicTokenizer(
        lowercase=True,
        remove_punct=False,
        remove_stopwords=False,
    )

    ds = load_dataset("LabHC/bias_in_bios")

    # Build vocabulary from the training split.
    counts = Counter()

    for text in tqdm(ds["train"]["hard_text"], desc="Building vocabulary"):
        tokens = to_vocab(
            embedding_name,
            tokenizer.tokens(text),
        )

        counts.update(tokens)

    # Retain words available in the pretrained embeddings.
    words = [
        word for word, _ in counts.most_common()
        if word in pretrained
    ]

    if max_vocab_size > 0:
        words = words[:max_vocab_size]

    word_to_idx = {
        word: idx + 2
        for idx, word in enumerate(words)
    }

    # Construct a compact embedding matrix.
    pretrained_vectors = pretrained.vectors

    if isinstance(pretrained_vectors, np.ndarray):
        pretrained_vectors = torch.from_numpy(pretrained_vectors)

    pretrained_vectors = pretrained_vectors.float()

    embedding_dim = pretrained_vectors.shape[1]

    vectors = [torch.zeros(embedding_dim)]

    # UNK vector: mean of all retained pretrained word vectors.
    if words:
        unk_vector = torch.stack([
            pretrained_vectors[pretrained.index(word)]
            for word in words
        ]).mean(dim=0)
    else:
        unk_vector = torch.zeros(embedding_dim)

    vectors.append(unk_vector)

    for word in tqdm(words, desc="Building embedding matrix"):
        vectors.append(
            pretrained_vectors[pretrained.index(word)]
        )

    embedding_matrix = torch.stack(vectors)

    def encode_split(split):
        sequences = []

        for text in tqdm(
            ds[split]["hard_text"],
            desc=f"Encoding {split}",
        ):
            tokens = to_vocab(
                embedding_name,
                tokenizer.tokens(text),
            )

            # Truncate long biographies.
            tokens = tokens[:max_length]

            # Unknown words map to UNK (index 1).
            ids = [
                word_to_idx.get(token, 1)
                for token in tokens
            ]

            # Avoid empty sequences.
            if not ids:
                ids = [1]

            sequences.append(ids)

        return BiographyDataset(
            sequences,
            ds[split]["profession"],
        )

    datasets = {
        split: encode_split(split)
        for split in ("train", "dev", "test")
    }

    return datasets, embedding_matrix, len(word_to_idx) + 2
