import itertools

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import csv
import os


class BinaryDistanceNN(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size, layers_num):
        super(BinaryDistanceNN, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(layers_num - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.3))
        layers.append(nn.Linear(hidden_dim, output_size))
        layers.append(nn.Sigmoid())
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)


def float_to_binary(value, num_bits):
    bits = []
    for _ in range(num_bits):
        value *= 2
        bit = int(value)
        bits.append(bit)
        value -= bit

    return torch.tensor(bits, dtype=torch.float32)


def binary_to_float(binary_tensor):
    binary_list = binary_tensor.int().tolist()
    float_value = 0.0
    for i, bit in enumerate(binary_list):
        if bit == 1:
            float_value += 2**(-i - 1)
    return float_value


def generate_dataset(num_samples, n, num_bits):
    X, y = [], []
    for _ in range(num_samples):
        vec1 = np.random.rand(n)
        vec2 = np.random.rand(n)
        distance = np.linalg.norm(vec1 - vec2) / np.sqrt(n)
        X.append(np.concatenate([vec1, vec2]))
        y.append(float_to_binary(distance, num_bits))
    return torch.tensor(X, dtype=torch.float32), torch.stack(y)


def train_and_evaluate_model(params, results_file):
    n = params['n']
    num_bits = params['num_bits']
    num_samples = params['num_samples']
    learning_rate = params['learning_rate']
    hidden_dim = params['hidden_dim']
    layers_num = params['layers_num']
    num_samples_test = params['num_samples_test']
    batch_size = params['batch_size']
    epochs = params['epochs']
    loss_tolerance = params['loss_tolerance']
    weight_decay = params['weight_decay']

    # Dane
    X_train, y_train = generate_dataset(num_samples, n, num_bits)
    X_val, y_val = generate_dataset(500, n, num_bits)
    X_test, y_test = generate_dataset(num_samples_test, n, num_bits)

    # Normalizacja
    X_train = F.normalize(X_train, p=2, dim=1)
    X_val = F.normalize(X_val, p=2, dim=1)
    X_test = F.normalize(X_test, p=2, dim=1)

    model = BinaryDistanceNN(input_size=2 * n, hidden_dim=hidden_dim, output_size=num_bits, layers_num=layers_num)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    early_stopping = False
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)

            optimizer.zero_grad()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            if loss.item() < loss_tolerance:
                early_stopping = True
                break
        if early_stopping:
            break


        # Walidacja
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                val_loss += criterion(model(batch_X), batch_y).item()
        val_loss /= len(val_loader)

        print(f"\rEpoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader):.6f}, Val Loss: {val_loss:.6f}")

    # Testowanie
    model.eval()
    with torch.no_grad():
        test_loss = criterion(model(X_test.to(device)), y_test.to(device)).item()
    print(f"Test Loss: {test_loss:.6f}")

    # Zapis wyników
    results = {**params, 'test_loss': test_loss}
    file_exists = os.path.isfile(results_file)
    with open(results_file, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=results.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(results)


if __name__ == '__main__':
    param_sets = []
    n_list = [15]
    num_bits_list = [8]
    num_samples_list = [1000, 2000, 3000, 4000, 5000]
    learning_rate_list = [0.001]
    hidden_dim_list = [16, 32]
    layers_num_list = [2, 3, 4]
    num_samples_test_list = [1000]
    epochs_list = [1000]
    batch_size_list = [64]

    loss_tolerance = 0.01
    weight_decay = 1e-4

    for n, num_bits, num_samples, learning_rate, hidden_dim, layers_num, num_samples_test, epochs, batch_size in itertools.product(
            n_list, num_bits_list, num_samples_list, learning_rate_list, hidden_dim_list, layers_num_list, num_samples_test_list, epochs_list, batch_size_list):
        param_sets.append({'n': n,
            'num_bits': num_bits,
            'num_samples': num_samples,
            'learning_rate': learning_rate,
            'hidden_dim': hidden_dim,
            'layers_num': layers_num,
            'num_samples_test': num_samples_test,
            'batch_size': batch_size,
            'epochs': epochs,
            'loss_tolerance': loss_tolerance,
            'weight_decay': weight_decay
            }
        )

    print(param_sets)

    results_file = "training_results.csv"

    for params in param_sets:
        train_and_evaluate_model(params, results_file)
