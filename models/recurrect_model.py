import torch
import torch.nn as nn
from data_generators import vector_generator as vg
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


class SimpleRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim_r, hidden_dim_fc=64, num_layers_recurrent=1, num_layers_fc=2):
        super(SimpleRNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim_r = hidden_dim_r
        self.hidden_dim_fc = hidden_dim_fc
        self.num_layers_recurrent = num_layers_recurrent
        self.num_layers_fc = num_layers_fc

        self.rnn = nn.LSTM(2, hidden_dim_r, num_layers_recurrent, batch_first=True)

        module_list = []
        for i in range(self.num_layers_fc):
            if i == 0:
                module_list.append(nn.Linear(hidden_dim_r, hidden_dim_fc))
            elif i != self.num_layers_fc - 1:
                module_list.append(nn.Linear(hidden_dim_fc, hidden_dim_fc))
            else:
                module_list.append(nn.Linear(hidden_dim_fc, 1))
        self.fc = nn.ModuleList(module_list)

    def forward(self, input):
        out, _ = self.rnn(input)
        out = out[:, -1, :]
        for layer in self.fc:
            out = layer(out)
        return out


class ExperimentalRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers_recurrent=1, num_layers_fc=2):
        super(ExperimentalRNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers_recurrent = num_layers_recurrent
        self.num_layers_fc = num_layers_fc

        self.rnn = nn.LSTM(2, hidden_dim, num_layers_recurrent, batch_first=True)

        inner_dim = 512
        module_list = []
        for i in range(self.num_layers_fc - 1):
            if i == 0:
                module_list.append(nn.Linear(hidden_dim, inner_dim))
            else:
                module_list.append(nn.Linear(inner_dim, inner_dim))
        module_list.append(nn.Linear(inner_dim, 1))
        self.fc = nn.ModuleList(module_list)

    def forward(self, input):
        out, _ = self.rnn(input)
        out = out[:, -1, :]
        for layer in self.fc:
            out = layer(out)
        return out


def train(model, input_size, num_epochs=100):
    n_samples = 64
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
    min_loss = float('inf')

    for epoch in range(num_epochs):
        x_data, y_data = vg.generate_sample_data_for_recurrent(n_samples, 0, 1, input_size)
        x_data, y_data = torch.tensor(x_data, dtype=torch.float).to(device), torch.tensor(y_data, dtype=torch.float).to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(x_data)

        # Compute the loss
        loss = criterion(output, y_data)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        # scheduler.step(loss.item())

        if loss.item() < 0.01:
            break

        if epoch % 1000 == 0:
            print(f'Epoch [{epoch}], Loss: {loss.item()}, lr={optimizer.param_groups[0]["lr"]}')

        if loss.item() < min_loss:
            min_loss = loss.item()
            print(f"Found new min_loss {min_loss} in epoch {epoch + 1}")

    return model


def validate(model, criterion, x_validate, y_validate):
    model.eval()
    with torch.no_grad():
        y_pred = model(x_validate)
        loss = criterion(y_pred, y_validate)
    for i in range(len(y_validate)):
        print("Predicted:", y_pred[i], "Actual:", y_validate[i])
    print("Mean loss:", loss.item())
    print("Max loss:", torch.max(abs(y_pred - y_validate)))
    print("Min loss:", torch.min(abs(y_pred - y_validate)))


if __name__ == '__main__':
    # Example usage:
    input_size = 75
    hidden_size = 128
    num_layers_recurrent = 2
    num_layers_fc = 3

    # Initialize SiameseRNN model
    model = ExperimentalRNN(input_size, hidden_size, num_layers_recurrent, num_layers_fc)

    # Train model
    model = train(model, input_size, num_epochs=50000)
    torch.save(model.state_dict(), "../saved_models/siamese_recurrent_model.pt")
    criterion = nn.L1Loss().to("cuda" if torch.cuda.is_available() else "cpu")

    x_data, y_data = vg.generate_sample_data_for_recurrent(100, 0, 1, input_size)
    x_data, y_data = torch.tensor(x_data, dtype=torch.float), torch.tensor(y_data, dtype=torch.float)

    validate(model, criterion, x_data, y_data)
