import torch
import torch.nn as nn
from data_generators import vector_generator as vg
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F


class SiameseLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(SiameseLSTM, self).__init__()
        self.lstm = nn.LSTM(1, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, hidden_size)

    def forward_once(self, x):
        # LSTM forward pass
        _, (hn, _) = self.lstm(x)
        # We take the last hidden state
        output = hn[-1]
        # Fully connected layer
        output = self.fc(output)
        return output

    def forward(self, x1, x2):
        output1 = self.forward_once(x1)
        output2 = self.forward_once(x2)
        return abs(output1 - output2)


def euclidean_distance(x1, x2):
    return F.pairwise_distance(x1, x2)


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_dist = euclidean_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_dist, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_dist, min=0.0), 2))
        return loss_contrastive


def train(model, input_size, num_epochs=100):
    n_samples = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train().to(device)

    # Define loss function and optimizer
    criterion = nn.L1Loss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    min_lr = 1e-8
    patience = 300
    patience_after_min_lr = 1000
    loops_after_min_lr = 0
    factor = 0.5
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=patience, factor=factor, verbose=True, min_lr=min_lr)

    for epoch in range(num_epochs):
        x_data, y_data = vg.generate_sample_data_for_recurrent(n_samples, 0, 1, input_size)
        x_data, y_data = torch.tensor(x_data, dtype=torch.float).to(device), torch.tensor(y_data, dtype=torch.float).to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(x_data[:, 0, :, :], x_data[:, 1, :, :])

        # Compute the loss
        loss = criterion(output, y_data)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        scheduler.step(loss.item())

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}], Loss: {loss.item()}, lr={optimizer.param_groups[0]["lr"]}')

    return model


if __name__ == '__main__':
    # Example usage:
    input_size = 4
    hidden_size = 20
    num_layers = 1

    # Generate random input tensors
    x_data, y_data = vg.generate_sample_data_for_recurrent(32, 0, 1, input_size)
    x_data, y_data = torch.tensor(x_data, dtype=torch.float), torch.tensor(y_data, dtype=torch.float)

    # Initialize SiameseRNN model
    model = SiameseLSTM(input_size, hidden_size, num_layers)

    # Train model
    model = train(model, input_size, num_epochs=10000)

    # Compute similarity score
    similarity_score = model(x_data[:, 0, :], x_data[:, 1, :])
    print("Out:", similarity_score, "Correct out:", y_data)
