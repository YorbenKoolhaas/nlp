from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

class RecurrentClassifier(nn.Module):
    """
    GRU/LSTM sequence classifier.

    Uses the final hidden state as the sequence representation.
    For bidirectional models, forward and backward states are
    concatenated.
    """

    def __init__(
        self,
        embedding_matrix,
        model=nn.GRU,
        hidden_size=128,
        num_layers=1,
        bidirectional=False,
        dropout=0.2,
        train_embeddings=False,
        projection_dim=0,
        num_classes=28,
    ):
        super().__init__()

        self.model_type = model
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix,
            freeze=not train_embeddings,
            padding_idx=0,
        )

        embedding_dim = embedding_matrix.shape[1]

        # Optional projection of pretrained embeddings.
        if projection_dim > 0:
            self.projection = nn.Linear(
                embedding_dim,
                projection_dim,
            )
            input_size = projection_dim
        else:
            self.projection = nn.Identity()
            input_size = embedding_dim

        self.encoder = model(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        output_size = hidden_size * (2 if bidirectional else 1)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(output_size, num_classes)

    def forward(self, tokens, lengths):
        x = self.embedding(tokens)
        x = self.projection(x)

        packed = pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        _, hidden = self.encoder(packed)

        # LSTM returns (hidden_state, cell_state).
        if isinstance(hidden, tuple):
            hidden = hidden[0]

        if self.bidirectional:
            # Last layer's forward and backward states.
            representation = torch.cat(
                (hidden[-2], hidden[-1]),
                dim=1,
            )
        else:
            representation = hidden[-1]

        representation = self.dropout(representation)

        return self.classifier(representation)
