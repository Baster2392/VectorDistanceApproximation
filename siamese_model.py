import torch
import torch.nn as nn


class SiameseNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SiameseNetwork, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # Neuron layers of model
        self.input_layer = ResizableInputLayer(input_dim)
        self.siamese_layer1 = FullyConnectedLayer(input_dim, hidden_dim)
        self.siamese_layer2 = FullyConnectedLayer(hidden_dim, hidden_dim)
        self.siamese_layer3 = FullyConnectedLayer(hidden_dim, hidden_dim)
        self.siamese_layer4 = FullyConnectedLayer(hidden_dim, hidden_dim)

        # Compare vectors
        self.fc = torch.norm

    def forward(self, x):
        # Split pairs
        x1 = self.input_layer.forward(x[:, 0, :])
        x2 = self.input_layer.forward(x[:, 1, :])

        x1 = self.siamese_layer1.forward(x1)
        x2 = self.siamese_layer1.forward(x2)



        out = torch.norm(x1 - x2, dim=1, keepdim=True)
        return out

    def scale_input_size(self, new_input_size):
        self.input_layer = ResizableInputLayer(new_input_size, self.hidden_dim)

    @staticmethod
    def loss_function(y_true, y_pred):
        return torch.norm(y_true - y_pred)


class FullyConnectedLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(FullyConnectedLayer, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(input_dim, hidden_dim))
        self.layer2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim))

    def forward(self, x):
        x = self.layer1(x)
        return x


class Conv1DLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Conv1DLayer, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1))

    def forward(self, x):
        x = self.layer1(x)
        return x


class SiameseBatchNorm1dLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SiameseBatchNorm1dLayer, self).__init__()
        self.layer1 = nn.Sequential(nn.BatchNorm1d(input_dim))

    def forward(self, x):
        x = self.layer1(x)
        return x


class ResizableInputLayer(nn.Module):
    def __init__(self, input_size):
        super(ResizableInputLayer, self).__init__()
        self.input_size = input_size

    def forward(self, x):
        return x

