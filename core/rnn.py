import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=64,
        hidden_size=64,
        num_layers=1,
        output_size=1,
        dropout=0.0,
    ):
        super().__init__()

        # Convert token IDs into learnable vectors.
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0,
        )

        # Process the embedded sequence.
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity="tanh",
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Predict the sentiment.
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch_size, sequence_length)

        # Convert token IDs into embeddings.
        embedded = self.embedding(x)
        # (batch_size, sequence_length, embedding_dim)

        # Process the sequence.
        _, hidden = self.rnn(embedded)
        # hidden: (num_layers, batch_size, hidden_size)

        # Extract the final layer's hidden state.
        final_hidden = hidden[-1]
        # (batch_size, hidden_size)

        # Produce one logit per review.
        logits = self.fc(final_hidden)
        # (batch_size, 1)

        return logits.squeeze(-1)
