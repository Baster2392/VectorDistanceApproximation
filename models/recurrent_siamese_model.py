import torch
import torch.nn as nn
from data_generators import vector_generator as vg
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


class SiameseRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(SiameseRNN, self).__init__()
        self.rnn = nn.LSTM(1, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(2 * hidden_size, 1)

    def forward(self, input1, input2):
        # Forward pass for first input
        out1, _ = self.rnn(input1)
        # Forward pass for second input
        out2, _ = self.rnn(input2)
        out1 = out1[:, -1, :]
        out2 = out2[:, -1, :]

        combined = torch.cat((out1, out2), 1)
        out = self.fc(combined)
        return out


def train(model, input_size, num_epochs=100):
    n_samples = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train().to(device)

    # Define loss function and optimizer
    criterion = nn.L1Loss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
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
    hidden_size = 128
    num_layers = 3

    # Generate random input tensors
    x_data, y_data = vg.generate_sample_data_for_recurrent(32, 0, 1, input_size)
    x_data, y_data = torch.tensor(x_data, dtype=torch.float), torch.tensor(y_data, dtype=torch.float)


    # Initialize SiameseRNN model
    model = SiameseRNN(input_size, hidden_size, num_layers)

    # Train model
    model = train(model, input_size, num_epochs=10000)

    # Compute similarity score
    similarity_score = model(x_data[:, 0, :], x_data[:, 1, :])
    print("Out:", similarity_score, "Correct out:", y_data)
