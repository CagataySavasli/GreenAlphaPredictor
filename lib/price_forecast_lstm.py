import torch
import torch.nn as nn
import numpy as np

class PriceForecastLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float = 0.2):
        """
        Initialize the LSTM model.

        Args:
            input_size (int): Number of input features.
            hidden_size (int): Number of hidden units.
            num_layers (int): Number of LSTM layers.
            dropout (float): Dropout rate to prevent overfitting.
        """
        super(PriceForecastLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)

        lstm_out, _ = self.lstm(x, (h0, c0))
        # Use the output from the last time step for prediction
        out = self.fc(lstm_out[:, -1, :])
        return out