import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import csv
import os

# Hierarchical Distance Representation Neural Network
class HierarchicalDistanceNN(nn.Module):
    def __init__(self, input_size, hidden_dim, coarse_bits, fine_bits, layers_num):
        super(HierarchicalDistanceNN, self).__init__()
        self.coarse_bits = coarse_bits
        self.fine_bits = fine_bits
        self.coarse_layer = self._build_layer(input_size, hidden_dim, coarse_bits, layers_num)
        self.fine_layer = self._build_layer(hidden_dim, hidden_dim, fine_bits, layers_num)

    def _build_layer(self, input_dim, hidden_dim, output_dim, layers_num):
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(layers_num - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)

    def forward(self, x):
        coarse_output = self.coarse_layer(x)
        fine_output = self.fine_layer(coarse_output)
        return coarse_output, fine_output

# Encode distance into hierarchical format
# coarse_bits: Coarse-grain precision (e.g., 4 bits for 16 intervals)
# fine_bits: Additional precision within each coarse interval
# e.g., 2 coarse bits + 4 fine bits: coarse interval [00,01,10,11], fine specifies within interval

def float_to_hierarchical(value, coarse_bits, fine_bits):
    max_value = 2 ** (coarse_bits + fine_bits) - 1
    int_value = int(value * max_value)
    coarse = (int_value >> fine_bits) & ((1 << coarse_bits) - 1)
    fine = int_value & ((1 << fine_bits) - 1)
    coarse_binary = [(coarse >> i) & 1 for i in range(coarse_bits)][::-1]
    fine_binary = [(fine >> i) & 1 for i in range(fine_bits)][::-1]
    return torch.tensor(coarse_binary + fine_binary, dtype=torch.float32)

# Dataset generator with hierarchical output

def generate_hierarchical_dataset(num_samples, n, coarse_bits, fine_bits):
    X, y = [], []
    for _ in range(num_samples):
        vec1 = np.random.rand(n)
        vec2 = np.random.rand(n)
        distance = np.linalg.norm(vec1 - vec2) / np.sqrt(n)
        X.append(np.concatenate([vec1, vec2]))
        y.append(float_to_hierarchical(distance, coarse_bits, fine_bits))
    return torch.tensor(X, dtype=torch.float32), torch.stack(y)

# Training and evaluation

def train_and_evaluate_model(params, results_file):
    n = params['n']
    coarse_bits = params['coarse_bits']
    fine_bits = params['fine_bits']
    num_samples = params['num_samples']
    num_samples_test = params['num_samples_test']
    learning_rate = params['learning_rate']
    hidden_dim = params['hidden_dim']
    layers_num = params['layers_num']
    batch_size = 32
    epochs = params['epochs']

    # Dataset preparation
    X_train, y_train = generate_hierarchical_dataset(num_samples, n, coarse_bits, fine_bits)
    X_val, y_val = generate_hierarchical_dataset(500, n, coarse_bits, fine_bits)
    X_test, y_test = generate_hierarchical_dataset(num_samples_test, n, coarse_bits, fine_bits)

    model = HierarchicalDistanceNN(input_size=2 * n, hidden_dim=hidden_dim, coarse_bits=coarse_bits, fine_bits=fine_bits, layers_num=layers_num)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # DataLoader setup
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            coarse_output, fine_output = model(batch_X)
            loss = criterion(coarse_output, batch_y[:, :coarse_bits]) + criterion(fine_output, batch_y[:, coarse_bits:])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_coarse_output, val_fine_output = model(X_val.to(device))
            val_loss = criterion(val_coarse_output, y_val[:, :coarse_bits].to(device)) + \
                       criterion(val_fine_output, y_val[:, coarse_bits:].to(device))
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader):.6f}, Val Loss: {val_loss:.6f}")

    # Test evaluation
    model.eval()
    with torch.no_grad():
        test_coarse_output, test_fine_output = model(X_test.to(device))
        test_loss = criterion(test_coarse_output, y_test[:, :coarse_bits].to(device)) + \
                    criterion(test_fine_output, y_test[:, coarse_bits:].to(device))
    print(f"Test Loss: {test_loss:.6f}")

    # Save results
    results = {**params, 'test_loss': test_loss.item()}
    file_exists = os.path.isfile(results_file)
    with open(results_file, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=results.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(results)

if __name__ == '__main__':
    param_sets = []
    n_list = [10]
    coarse_bits_list = [4]
    fine_bits_list = [4]
    num_samples_list = [10000]
    learning_rate_list = [0.001]
    hidden_dim_list = [128]
    layers_num_list = [3]
    num_samples_test_list = [1000]
    epochs_list = [50]

    for n, coarse_bits, fine_bits, num_samples, num_samples_test, learning_rate, hidden_dim, layers_num, epochs in itertools.product(
            n_list, coarse_bits_list, fine_bits_list, num_samples_list, num_samples_test_list, learning_rate_list, hidden_dim_list, layers_num_list, epochs_list):
        param_sets.append({'n': n,
                           'coarse_bits': coarse_bits,
                           'fine_bits': fine_bits,
                           'num_samples': num_samples,
                           'learning_rate': learning_rate,
                           'hidden_dim': hidden_dim,
                           'layers_num': layers_num,
                           'num_samples_test': num_samples_test,
                           'epochs': epochs})

    results_file = "hierarchical_results.csv"
    for params in param_sets:
        train_and_evaluate_model(params, results_file)
