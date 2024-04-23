import torch
import torch.nn as nn


class SiameseNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SiameseNetwork, self).__init__()

        self.hidden_dim = hidden_dim
        # Neuron layers of model
        self.input_layer = ResizableInputLayer(input_dim, hidden_dim)
        self.siamese_layer = SiameseLayer(hidden_dim, hidden_dim)

        # Compare vectors
        self.fc = torch.linalg.norm

    def forward(self, x):
        # Split pairs
        x1 = self.input_layer.forward(x[:, 0, :])
        x2 = self.input_layer.forward(x[:, 1, :])

        x1 = self.siamese_layer.forward(x1)
        x2 = self.siamese_layer.forward(x2)

        # Propagacja przez warstwę przekształcającą
        out = torch.linalg.norm(x1 - x2, dim=1, keepdim=True)
        return out

    def scale_input_size(self, new_input_size):
        self.input_layer = ResizableInputLayer(new_input_size, self.hidden_dim)


class ResizableInputLayer(nn.Module):
    def __init__(self, input_size, hidden_dim):
        super(ResizableInputLayer, self).__init__()
        self.input_size = input_size
        # Neurons with constant weights
        self.weights = nn.Parameter(torch.eye(input_size, hidden_dim) / input_size, requires_grad=False)

    def forward(self, x):
        return torch.matmul(x, self.weights)


class SiameseLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SiameseLayer, self).__init__()
        self.neurons_layer1 = nn.Sequential(nn.Linear(input_dim, hidden_dim))
        self.neurons_layer2 = nn.Sequential(nn.Linear(input_dim, hidden_dim))
        self.neurons_layer3 = nn.Sequential(nn.Linear(input_dim, hidden_dim))

    def forward(self, x):
        x = self.neurons_layer1(x)
        return x
