import math
import numpy as np
import torch
from torch import nn
import time
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
time_start = time.time()
FILENAME = f'./data_demnd_results/data_demand_{time_start}.csv'

def subtract_vectors_torch(v1, v2):
    return v1 - v2

def square_torch(v):
    return torch.pow(v, 2)

def sum_torch(v, dim):
    return torch.sum(v, dim=dim)

def sqrt_torch(v):
    return torch.sqrt(v)

def subtract_vectors_np(v1, v2):
    return v1 - v2

def square_np(v):
    return np.power(v, 2)

def sum_np(v, axis):
    return np.sum(v, axis=axis)

def sqrt_np(v):
    return np.sqrt(v)

def calculate_euclidean_distance_torch(x_dataset):
    v1, v2 = x_dataset[:, :, 0], x_dataset[:, :, 1]
    diff = subtract_vectors_torch(v1, v2)
    squared = square_torch(diff)
    summed = sum_torch(squared, dim=1)
    return sqrt_torch(summed).to(device)

def calculate_euclidean_distance_np(x_dataset):
    v1, v2 = x_dataset[:, :, 0], x_dataset[:, :, 1]
    diff = subtract_vectors_np(v1, v2)
    squared = square_np(diff)
    summed = sum_np(squared, axis=1)
    return sqrt_np(summed)

def calculate_distance(x_dataset, metric='euclidean', use_analytical=True):
    if metric == 'euclidean':
        if use_analytical:
            x_dataset_np = x_dataset.cpu().numpy() if isinstance(x_dataset, torch.Tensor) else x_dataset
            return torch.tensor(calculate_euclidean_distance_np(x_dataset_np), device=device)
        else:
            return calculate_euclidean_distance_torch(x_dataset)

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim_r, hidden_dim_fc=64, num_layers_recurrent=1, num_layers_fc=2):
        super(LSTMModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim_r = hidden_dim_r
        self.hidden_dim_fc = hidden_dim_fc
        self.num_layers_recurrent = num_layers_recurrent
        self.num_layers_fc = num_layers_fc

        self.rnn = nn.LSTM(2, hidden_dim_r, num_layers_recurrent, batch_first=True)

        module_list = []
        if self.num_layers_recurrent == 1:
            module_list.append(nn.Linear(self.hidden_dim_r, 1))
        else:
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
        for i in range(self.num_layers_fc - 1):
            out = nn.functional.leaky_relu(self.fc[i](out))
        return nn.functional.sigmoid(self.fc[-1](out))

def generate_vectors(vectors_number, vector_size):
    return torch.rand((vectors_number, vector_size), dtype=torch.float32, device=device)

def train(model, input_dim, optimizer, criterion, max_epochs, data_demand, factor, metric, loss_tolerance=0.01, batch_size=64, mode='rnn', use_analytical=False):
    max_distance = math.sqrt(input_dim)
    print("Training model for parameters:")
    print("input_dim:", input_dim)
    print("hidden_dim_r:", model.hidden_dim_r)
    print("hidden_dim_fc:", model.hidden_dim_fc)
    print("num_layers_recurrent:", model.num_layers_recurrent)
    print("num_layers_fc:", model.num_layers_fc)
    print("data_demand:", data_demand)
    print("max_distance:", max_distance)
    print("factor", factor)
    print("loss_tolerance:", loss_tolerance)
    print("batch_size:", batch_size)

    best_loss = float("inf")
    best_loss_epoch = 0
    dataset = generate_vectors(data_demand, input_dim)

    model.train()
    for epoch in range(max_epochs):
        pairs_indexes = np.array([np.random.choice(len(dataset), 2, replace=False) for _ in range(batch_size)])
        x_data = dataset[pairs_indexes]
        if mode == 'rnn':
            x_data = x_data.permute(0, 2, 1)

        y_data = calculate_distance(x_data, metric=metric, use_analytical=use_analytical).unsqueeze(1) / max_distance

        optimizer.zero_grad()
        output = model(x_data)
        loss = criterion(output, y_data)
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_loss_epoch = epoch

        print(f'\rEpoch: {epoch}/{max_epochs}, Loss: {loss.item():.4f}, Best loss: {best_loss}, Best loss as distance(best_loss * max_distance): {best_loss * max_distance}, Best loss epoch: {best_loss_epoch}', end='')

        if loss.item() < loss_tolerance:
            print()
            print('Stopping training...')
            return model, epoch, loss.item()
    return model, max_epochs, loss.item()

def test(model, input_dim, criterion, test_dataset_size, metric, mode='rnn', use_analytical=False):
    print("\nTesting model...")
    model.eval()
    with torch.no_grad():
        dataset = generate_vectors(test_dataset_size, input_dim)
        pairs_indexes = np.array([np.random.choice(len(dataset), 2, replace=False) for _ in range(test_dataset_size)])
        x_data = dataset[pairs_indexes]
        if mode == 'rnn':
            x_data = x_data.permute(0, 2, 1)
        y_data = calculate_distance(x_data, metric=metric, use_analytical=use_analytical).unsqueeze(-1) / math.sqrt(input_dim)
        output = model(x_data)
        loss = criterion(output, y_data)

        for i in range(20):
            print(f'Predicted: {output[i].item()}, Actual: {y_data[i].item()}')
    return loss

def search_data_demand_recurrent():
    print("Using device:", device)
    tests_number = 5
    extrapolated_data_demand = [10000, 10000, 10000, 10000]
    test_dataset_size = 1000
    hidden_dim_r = 64
    hidden_dim_fc = 700
    num_layers_recurrent = 2
    num_layers_fc = 3
    metric = 'euclidean'

    input_dims = [800]
    loss_tolerance = 0.01
    lr = 0.001
    criterion = nn.L1Loss().to(device)

    # dataset.length = extrapolated_data_demand * factor
    min_factor = 1.0
    max_factor = 1.0
    step = 1.0
    max_epochs = 1000000

    loops = math.ceil((max_factor - min_factor) / step)

    for _ in range(tests_number):
        for input_dim in input_dims:
            current_loss_tolerance = loss_tolerance / math.sqrt(input_dim)
            for i in range(loops + 1):
                model = LSTMModel(input_dim, hidden_dim_r, hidden_dim_fc, num_layers_recurrent, num_layers_fc).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                data_demand = extrapolated_data_demand[input_dims.index(input_dim)]
                factor = min_factor + i * step
                data_demand = math.floor(data_demand * factor)
                model, epoch, train_loss = train(model, input_dim, optimizer, criterion, max_epochs, data_demand, factor, metric, loss_tolerance=current_loss_tolerance, batch_size=64)

                test_loss = test(model, input_dim, criterion, test_dataset_size, metric)
                max_distance = math.sqrt(input_dim)
                train_loss_distance = train_loss * max_distance
                test_loss_distance = test_loss * max_distance
                file_exists = os.path.isfile(FILENAME)
                with open(FILENAME, 'a') as file:
                    if not file_exists:
                        file.write("Input_dim,Hidden_dim_r,Hidden_dim_fc,Num_layers_recurrent,Num_layers_fc,Loss_tolerance,Data_demand,Dd_factor,Train_loss,Test_loss,Train_loss_distance,Test_loss_distance,Epochs\n")
                    file.write(f"{model.input_dim},{model.hidden_dim_r},{model.hidden_dim_fc},{model.num_layers_recurrent},{model.num_layers_fc},{loss_tolerance},{data_demand},{factor},{train_loss},{test_loss},{train_loss_distance},{test_loss_distance},{epoch}\n")

if __name__ == '__main__':
    search_data_demand_recurrent()
