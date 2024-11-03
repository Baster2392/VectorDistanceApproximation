import torch
import torch.nn as nn
import torch.nn.functional as F


"""
The model now take's in a metric argument, which either enables or disables scaling of the output. 
See our paper for proper explanation - the structure of the Network is explained there as well.
"""


class SiameseNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_siamese_layers, num_shared_layers):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_shared_layers = num_shared_layers
        self.num_siamese_layers = num_siamese_layers

        super(SiameseNetwork, self).__init__()
        #self.dropout = nn.Dropout(p=0.1)

        self.siamese_layers = nn.ModuleList(
            [nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim) for i in range(num_siamese_layers)])

        shared_layers = []
        for _ in range(num_shared_layers):
            shared_layers.append(nn.Linear(2*hidden_dim, 2*hidden_dim))
            shared_layers.append(nn.ReLU())

        shared_layers.append(nn.Linear(2*hidden_dim, 1))
        self.shared_layers = nn.Sequential(*shared_layers)

    def forward(self, x, metric="custom_metric"):
        scaling_factor = torch.max(torch.abs(x))
        x = x / scaling_factor
        x1, x2 = x[:, 0, :], x[:, 1, :]
        # x1 = self.dropout(x1)
        # x2 = self.dropout(x2)
        for layer in self.siamese_layers:
            x1 = F.relu(layer(x1))
            x2 = F.relu(layer(x2))
        concatenated = torch.cat((x1, x2), dim=1)
        output = self.shared_layers(concatenated)
        if metric != 'cosine':
            output = output * scaling_factor
        return output

