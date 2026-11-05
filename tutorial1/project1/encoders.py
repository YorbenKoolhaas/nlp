## token indicators
import numpy as np
from collections import Counter
from tqdm import tqdm

from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Metaspace
from tokenizers.trainers import BpeTrainer
from tokenizers.models import BPE


def train_tokenizer(corpus, vocab_size, min_frequency, pretokenizer=Metaspace(), unk_token="[UNK]"):
    """
    corpus: a list of texts (each a string)
    vocab_size: number of distinct BPE tokens
    min_frequency: we only merge token pairs that are frequent enough
    pretokenizer: a simple tokenizer that we use before BPE tokenization
    
        You probably don't need to change the pretokenizer, unless you are working with
        languages that really don't use any form of blank space 
        (then you could, for example, pretokenize at character level)
    """
    # Create an empty BPE tokenizer
    tokenizer = Tokenizer(BPE(unk_token=unk_token)) #unk_token used for characters that cannot be encoded

    # First split the corpus into words
    tokenizer.pre_tokenizer = pretokenizer

    # Tell BPE how large the vocabulary should be
    trainer = BpeTrainer(
        vocab_size=vocab_size, # The size of the vocabulary to learn.
        min_frequency=min_frequency, # Only perform a merge if the pair occurs at least min_frequency times in the training corpus.
        special_tokens=[unk_token]
    )

    # Learn the vocabulary and merge rules
    tokenizer.train_from_iterator(corpus, trainer=trainer)

    return tokenizer


class Encoder:

    def __len__(self) -> int:
        """Dimensionality of the representation space"""
        raise NotImplementedError("Concrete implementations will be discussed later")

    def __call__(self, x: str) -> 'np.array':
        """Represent x as a D-dimensional vector of real numbers"""
        raise NotImplementedError("Concrete implementations will be discussed later")


class TokenIndicator(Encoder):

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        
    def __len__(self) -> int:
        """Dimensionality of the representation space"""
        return len(self.tokenizer.get_vocab())

    def __call__(self, x: str) -> 'np.array':
        """Represent x as a D-dimensional vector of real numbers"""
        unique_token_ids = list(set(self.tokenizer.encode(x).ids))

        # A 0 vector with the same length as the vocabulary except in positions corresponding 
        # to tokens appearing in the document
        vector = np.zeros(len(self))
        vector[unique_token_ids] = 1

        return vector


class TfIdfEncoder():

    def __init__(self, tokenizer: Tokenizer, corpus: list):
        self.tokenizer = tokenizer

        # Building the document frequency (df) dictionary
        df = Counter()
        for doc in tqdm(corpus):
            for token_id in set(self.tokenizer.encode(doc).ids):
                df[token_id] += 1

        # Calculating the inverse document frequency (idf) for each token
        N = len(corpus)
        self.idf = {token_id: np.log(N / df[token_id]) for token_id in df}


    def __len__(self) -> int:
        """Dimensionality of the representation space"""
        return len(self.tokenizer.get_vocab())

    def __call__(self, x: str) -> 'np.array':
        """Represent x as a D-dimensional vector of real numbers"""
        x_encoded = self.tokenizer.encode(x).ids

        counts = Counter(x_encoded)
        vector = np.zeros(len(self.tokenizer.get_vocab()))

        for token_id, count in counts.items():
            tf = count / len(x_encoded)  # term frequency
            vector[token_id] = tf * self.idf.get(token_id)

        # L2 normalisation: every document vector gets length 1 (as in sklearn's TfidfVectorizer).
        # Without it the values are tiny (~0.01-0.05), which regularisation in loglin/MLP pushes towards zero.
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm

        return vector


class BagOfWords(Encoder):

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        """Dimensionality of the representation space"""
        return len(self.tokenizer.get_vocab())

    def __call__(self, x: str) -> 'np.array':
        """Represent x as a D-dimensional vector of real numbers"""
        unique_token_ids = Counter(self.tokenizer.encode(x).ids)

        vector = np.zeros(len(self))
        for id, count in unique_token_ids.items():
            vector[id] = count
        return vector
