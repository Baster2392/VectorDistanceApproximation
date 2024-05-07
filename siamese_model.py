import torch
import torch.nn as nn


class SiameseNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=4):
        super(SiameseNetwork, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Neuron layers of model
        self.input_layer = ResizableInputLayer(input_dim)

        # Create siamese layers dynamically
        self.siamese_layers = nn.ModuleList([
            FullyConnectedLayer(input_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])

        # Compare vectors
        self.fc = torch.norm

    def forward(self, x):
        # Split pairs
        x1 = self.input_layer.forward(x[:, 0, :])
        x2 = self.input_layer.forward(x[:, 1, :])

        for siamese_layer in self.siamese_layers:
            x1 = siamese_layer.forward(x1)
            x2 = siamese_layer.forward(x2)

        out = torch.norm(x1 - x2, dim=1, keepdim=True)
        return out

    def scale_input_size(self, new_input_size):
        self.input_layer = ResizableInputLayer(new_input_size)


class FullyConnectedLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(FullyConnectedLayer, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(input_dim, hidden_dim))
        self.layer2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim))

    def forward(self, x):
        x = self.layer1(x)
        return x

class ResizableInputLayer(nn.Module):
    def __init__(self, input_size):
        super(ResizableInputLayer, self).__init__()
        self.input_size = input_size

    def forward(self, x):
        return x
