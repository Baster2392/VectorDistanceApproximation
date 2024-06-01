import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, num_shared_layers=1):
        super(SiameseNetwork, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_shared_layers = num_shared_layers

        # Create siamese layers dynamically
        self.siamese_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim),
                nn.ReLU()
            )
            for i in range(num_layers)
        ])

        # Create shared layers dynamically
        shared_layers = [nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()) for _ in range(num_shared_layers - 1)]
        shared_layers.append(nn.Linear(hidden_dim, 1))
        self.shared_layers = nn.ModuleList(shared_layers)

    def forward(self, x):
        # Split pairs
        x1 = x[:, 0, :]
        x2 = x[:, 1, :]

        #x1 = torch.log(abs(x1+1e-5))
        #x2 = torch.log(abs(x2+1e-5))

        # Forward pass through siamese layers
        for siamese_layer in self.siamese_layers:
            x1 = siamese_layer(x1)
            x2 = siamese_layer(x2)

        # Combine outputs
        combined_x = torch.abs(x1 - x2)

        # Pass combined output through shared layers
        for shared_layer in self.shared_layers:
            combined_x = shared_layer(combined_x)   # Last layer returns distance

        return combined_x


