import torch
import torch.nn as nn
import torch.optim as optim
import old.data_generators.reduced_vectors_generator as vg


class DimensionalReductionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, layers=3):
        super(DimensionalReductionModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.layers = layers

        self.layers_list = nn.ModuleList()
        for i in range(layers):
            if i == 0:
                self.layers_list.append(nn.Linear(self.input_size, self.hidden_size))
            elif i == layers - 1:
                self.layers_list.append(nn.Linear(self.hidden_size, self.output_size))
            else:
                self.layers_list.append(nn.Linear(self.hidden_size, self.hidden_size))

    def forward(self, x):
        for layer in self.layers_list:
            x = layer(x)
        return x


def train_model(model):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    size = model.input_size
    batch_size = 256
    min_loss = float('inf')

    max_epochs = 500000
    for epoch in range(max_epochs):
        data_x, data_y = vg.generate_sample_data(size, 0, 1, batch_size)
        data_x = torch.tensor(data_x, dtype=torch.float)
        data_y = torch.tensor(data_y, dtype=torch.float)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(data_x)

        # Compute the loss
        loss = criterion(output, data_y)

        # Backward propagation and optimization
        loss.backward()
        optimizer.step()

        if loss.item() < 0.01:
            break

        if epoch % 100 == 0:
            print(f'Epoch [{epoch}], Loss: {loss.item()}, lr={optimizer.param_groups[0]["lr"]}')

        if loss.item() < min_loss:
            min_loss = loss.item()
            print(f"Found new min_loss {min_loss} in epoch {epoch + 1}")

    return model


if __name__ == '__main__':
    model = DimensionalReductionModel(input_size=20, hidden_size=64, output_size=2, layers=3)
    model = train_model(model)
