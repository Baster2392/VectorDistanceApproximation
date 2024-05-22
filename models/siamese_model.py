import torch
import torch.nn as nn


class SiameseNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(SiameseNetwork, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Create siamese layers dynamically
        self.siamese_layers = nn.ModuleList([
            nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])

        # Compare vectors
        self.shared_layer = nn.ModuleList(
            [
                nn.Linear(hidden_dim, input_dim),
                nn.Linear(input_dim, 1)
            ]
        )

    def forward(self, x):
        # Split pairs
        x1 = x[:, 0, :]
        x2 = x[:, 1, :]

        # Forward pass through siamese layer
        for siamese_layer in self.siamese_layers:
            x1 = siamese_layer.forward(x1)
            x2 = siamese_layer.forward(x2)

        # Combine outputs
        combined_x = abs(x1 - x2)

        # Pass combined output through shared fc layers
        for shared_layer in self.shared_layer:
            combined_x = shared_layer.forward(combined_x)   # Last layer returns distance

        return combined_x
