import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(
        self,
        input_size=1,
        hidden_size=16,
        num_layers=1,
        output_size=1
    ):
        super().__init__()

        # Recurrent layer
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # Prediction layer
        self.fc = nn.Linear(
            hidden_size,
            output_size
        )

    def forward(self, x):
        # Process the input sequence
        output, hidden = self.rnn(x)

        # Use the final timestep's output
        last_output = output[:, -1, :]

        # Predict the next value
        prediction = self.fc(last_output)

        return prediction
