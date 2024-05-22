import torch
import torch.nn as nn
from data_generators import vector_generator as vg
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


class SiameseRNN(nn.Module):
    def __init__(self, hidden_dim, num_layers=1):
        super(SiameseRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.rnn = nn.LSTM(1, hidden_dim, num_layers, batch_first=True)

    def forward_once(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        return out

    def forward(self, input1, input2):
        input1 = torch.log(input1 + 1e-6)
        input2 = torch.log(input2 + 1e-6)

        out1 = self.forward_once(input1)
        out2 = self.forward_once(input2)

        # Compute the Euclidean distance between the two outputs
        distance = torch.norm(out1 - out2, dim=1)
        return distance


def train(model, input_size, num_epochs=100):
    n_samples = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train().to(device)

    # Define loss function and optimizer
    criterion = nn.L1Loss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    min_lr = 1e-8
    patience = 500
    factor = 0.5
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=patience, factor=factor, min_lr=min_lr)

    for epoch in range(num_epochs):
        x_data, y_data = vg.generate_sample_data_for_recurrent_siamese(n_samples, 0, 1, input_size, input_size + 1)
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

        if loss.item() < 0.1:
            break

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}], Loss: {loss.item()}, lr={optimizer.param_groups[0]["lr"]}')

    return model


def validate(model, criterion, x_validate1, x_validate2, y_validate):
    model.eval()
    with torch.no_grad():
        y_pred = model(x_validate1, x_validate2)
        loss = criterion(y_pred, y_validate)
    for i in range(len(y_validate)):
        print("Predicted:", y_pred[i], "Actual:", y_validate[i])
    print("Mean loss:", loss.item())
    print("Max loss:", torch.max(abs(y_pred - y_validate)))
    print("Min loss:", torch.min(abs(y_pred - y_validate)))


if __name__ == '__main__':
    # Example usage:
    input_size = 5
    hidden_size = 256
    num_layers = 3

    # Initialize SiameseRNN model
    model = SiameseRNN(hidden_size, num_layers)

    # Train model
    model = train(model, input_size, num_epochs=10000)
    torch.save(model.state_dict(), "../saved_models/siamese_recurrent_model.pt")
    criterion = nn.L1Loss().to("cuda" if torch.cuda.is_available() else "cpu")

    x_data, y_data = vg.generate_sample_data_for_recurrent_siamese(32, 0, 1, input_size, input_size + 1)
    x_data, y_data = torch.tensor(x_data, dtype=torch.float), torch.tensor(y_data, dtype=torch.float)

    validate(model, criterion, x_data[:, 0, :, :], x_data[:, 1, :, :], y_data)
