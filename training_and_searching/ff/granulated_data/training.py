import csv
import itertools
import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split


class DistanceNN(nn.Module):
    def __init__(self, input_size, hidden_dim, layers_num):
        super(DistanceNN, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(layers_num - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.3))
        layers.append(nn.Linear(hidden_dim, 1))
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)


def generate_dataset(n, granulation=2, granulation_mode=True, dataset_size=1000, epsilon=0.0):
    if granulation_mode:
        step = 1 / (granulation - 1)
        values = [round(i * step, 10) for i in range(granulation)]
        vectors = torch.tensor(list(itertools.product(values, repeat=n)), dtype=torch.float32)

        if epsilon != 0:
            vectors = vectors + (2 * torch.randn(vectors.shape) - 1) * epsilon
    else:
        vectors = torch.randn((dataset_size, n), dtype=torch.float32)
        vectors = (vectors - vectors.min()) / (vectors.max() - vectors.min())
    return vectors


def calculate_distance(p1, p2):
    return torch.sqrt(((p1 - p2) ** 2).sum(dim=-1))


def train_and_evaluate_model(params, end_results_file):
    n = params['n']
    learning_rate = params['learning_rate']
    hidden_dim = params['hidden_dim']
    layers_num = params['layers_num']
    batch_size = params['batch_size']
    epochs = params['epochs']
    granulation = params['granulation']
    weight_decay = params['weight_decay']
    loss_tolerance = params['loss_tolerance']
    dataset_size = params['dataset_size']
    granulation_mode = params['granulation_mode']
    epsilon = params['epsilon']
    progress_results_filename = (
                str(n) + '_' + str(hidden_dim) + '_' + str(layers_num) + '_' + str(epochs) + '_' + str(granulation) +
                '_' + str(loss_tolerance) + '_' + str(granulation_mode) + '_' + str(epsilon) + '.csv')

    print(f'Training for: n={n}, hidden_dim={hidden_dim}, layers_num={layers_num},'
          f'granulation={granulation}, granulation_mode={granulation_mode}, epsilon={epsilon}')

    # Data
    X = generate_dataset(n, granulation=granulation, granulation_mode=granulation_mode, dataset_size=dataset_size, epsilon=epsilon)
    X_val_normal = generate_dataset(n, granulation_mode=False, dataset_size=int(dataset_size * 0.1), epsilon=epsilon)
    lengths = [int(0.8 * len(X)), int(0.1 * len(X)), len(X) - int(0.8 * len(X)) - int(0.1 * len(X))]
    X_train, X_val, X_test = random_split(X, lengths)

    # Normalization by X_train
    mean = X_train.dataset.mean(dim=0)
    std = X_train.dataset.std(dim=0)
    X_train = (X_train.dataset - mean) / std
    X_val = (X_val.dataset - mean) / std
    X_test = (X_test.dataset - mean) / std
    X_val_normal = (X_val_normal - mean) / std

    train_loader = DataLoader(TensorDataset(X_train), batch_size=batch_size * 2, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val), batch_size=batch_size * 2, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test), batch_size=batch_size * 2, shuffle=False)
    val_normal_loader = DataLoader(TensorDataset(X_val_normal), batch_size=batch_size * 2, shuffle=False)

    model = DistanceNN(input_size=2 * n, hidden_dim=hidden_dim, layers_num=layers_num)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Training
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        train_batches = 0
        for batch_X in train_loader:
            if batch_X[0].size(0) % 2 != 0:
                batch_X = (batch_X[0][:-1],)

            p1, p2 = batch_X[0].chunk(2, dim=0)
            if p1.size(0) != p2.size(0):
                continue

            batch_X = torch.cat((p1, p2), dim=1).to(device)
            batch_y = calculate_distance(p1, p2).to(device)

            outputs = model(batch_X)
            optimizer.zero_grad()
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            train_batches += 1

        train_loss = epoch_loss / train_batches if train_batches > 0 else float('inf')

        # Evaluate model
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch_X in val_loader:
                if batch_X[0].size(0) % 2 != 0:
                    batch_X = (batch_X[0][:-1],)

                p1, p2 = batch_X[0].chunk(2, dim=0)
                if p1.size(0) != p2.size(0):
                    continue

                batch_X = torch.cat((p1, p2), dim=1).to(device)
                batch_y = calculate_distance(p1, p2).to(device)
                val_loss += criterion(model(batch_X).squeeze(), batch_y).item()
                val_batches += 1

        val_loss = val_loss / val_batches if val_batches > 0 else float('inf')

        # Evaluate model using data with normal distribution
        val_normal_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch_X in val_normal_loader:
                if batch_X[0].size(0) % 2 != 0:
                    batch_X = (batch_X[0][:-1],)

                p1, p2 = batch_X[0].chunk(2, dim=0)
                if p1.size(0) != p2.size(0):
                    continue

                batch_X = torch.cat((p1, p2), dim=1).to(device)
                batch_y = calculate_distance(p1, p2).to(device)
                val_normal_loss += criterion(model(batch_X).squeeze(), batch_y).item()
                val_batches += 1

        val_normal_loss = val_normal_loss / val_batches if val_batches > 0 else float('inf')

        print(
            f'\rEpoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Val Normal Loss: {val_normal_loss:.6f}',
            end='')

        # Save training results
        results = {'epoch': epoch + 1, 'train_loss': train_loss, 'val_loss': val_loss, 'val_normal_loss': val_normal_loss}
        file_exists = os.path.isfile('progress_results/' + progress_results_filename)
        with open('progress_results/' + progress_results_filename, 'a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=results.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(results)

        if val_loss < loss_tolerance:
            break

    test_loss = 0.0
    test_batches = 0
    model.eval()
    with torch.no_grad():
        for batch_X in test_loader:
            if batch_X[0].size(0) % 2 != 0:
                batch_X = (batch_X[0][:-1],)

            p1, p2 = batch_X[0].chunk(2, dim=0)
            if p1.size(0) != p2.size(0):
                continue

            batch_X = torch.cat((p1, p2), dim=1).to(device)
            batch_y = calculate_distance(p1, p2).to(device)
            outputs = model(batch_X)
            test_loss += criterion(outputs.squeeze(), batch_y).item()
            test_batches += 1

            for i in range(len(outputs)):
                print(f'Predicted: {outputs[i]}, Actual: {batch_y[i]}')

    test_loss = test_loss / test_batches if test_batches > 0 else float('inf')
    print(f"Test Loss: {test_loss:.6f}")

    test_loss = 0.0
    test_batches = 0
    model.eval()
    with torch.no_grad():
        for batch_X in val_normal_loader:
            if batch_X[0].size(0) % 2 != 0:
                batch_X = (batch_X[0][:-1],)

            p1, p2 = batch_X[0].chunk(2, dim=0)
            if p1.size(0) != p2.size(0):
                continue

            batch_X = torch.cat((p1, p2), dim=1).to(device)
            batch_y = calculate_distance(p1, p2).to(device)
            outputs = model(batch_X)
            test_loss += criterion(outputs.squeeze(), batch_y).item()
            test_batches += 1

    test_normal_loss = test_loss / test_batches if test_batches > 0 else float('inf')
    print(f"Test Normal Loss: {test_normal_loss:.6f}")

    results = {**params, 'test_loss': test_loss, 'test_normal_loss': test_normal_loss}
    file_exists = os.path.isfile('end_results/' + end_results_file)
    with open('end_results/' + end_results_file, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=results.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(results)


if __name__ == '__main__':
    # prepare parameters lists
    param_sets = []
    n_list = [5, 6, 7]
    learning_rate_list = [0.01]
    hidden_dim_list = [32]
    layers_num_list = [3]
    epochs_list = [100]
    batch_size_list = [64]
    granulation_list = [2, 3, 4, 5]   # 0 means normal distribution
    granulation_mode_list = [False, True]
    epsilon_list = [0.0, 0.05, 0.1, 0.2]

    weight_decay = 1e-4
    loss_tolerance = 0.00

    for n, learning_rate, hidden_dim, layers_num, epochs, batch_size, granulation, granulation_mode, epsilon in itertools.product(
            n_list, learning_rate_list, hidden_dim_list, layers_num_list, epochs_list, batch_size_list,
            granulation_list, granulation_mode_list, epsilon_list):
        if not granulation_mode and epsilon != 0.0:
            continue
        param_sets.append({'n': n,
                           'learning_rate': learning_rate,
                           'hidden_dim': hidden_dim,
                           'layers_num': layers_num,
                           'batch_size': batch_size,
                           'epochs': epochs,
                           'weight_decay': weight_decay,
                           'granulation': granulation,
                           'loss_tolerance': loss_tolerance,
                           'dataset_size': n**(granulation + 1),
                           'granulation_mode': granulation_mode,
                           'epsilon': epsilon
                           }
                          )

    results_file = "training_results_with_normal_distribution_epsilon.csv"

    for params in param_sets:
        train_and_evaluate_model(params, results_file)
