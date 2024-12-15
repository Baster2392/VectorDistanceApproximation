import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
import random
import csv


# Generate dataset
def generate_dataset(num_samples, input_dim):
    print(f"Generating data for id: {input_dim}")
    X1 = np.random.uniform(-1, 1, (num_samples, input_dim))
    X2 = np.random.uniform(-1, 1, (num_samples, input_dim))

    Y = np.sqrt(np.sum((X2 - X1) ** 2, axis=1))

    X = np.hstack((X1, X2))

    return X, Y


# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')


# Define the neural network
class EuclideanDistanceNN(nn.Module):
    def __init__(self, layer_sizes, input_dim):
        super(EuclideanDistanceNN, self).__init__()
        layers = []
        input_size = 2 * input_dim
        for size in layer_sizes:
            layers.append(nn.Linear(input_size, size))
            layers.append(nn.ReLU6())
            input_size = size
        layers.append(nn.Linear(input_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        scaling_factor = torch.max(torch.abs(x))
        x = x / scaling_factor
        x = self.network(x)
        return x * scaling_factor


# Function to train and evaluate the model
def train_and_evaluate_model(input_dim, layer_sizes, learning_rate, weight_decay, num_epochs=100000, batch_size=64,
                             early_stopping_threshold=0.1):
    model = EuclideanDistanceNN(layer_sizes, input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)
    best_model_state = None

    num_batches = X_train.size()[0] // batch_size
    min_loss = float('inf')
    epochs = 0
    for epoch in range(num_epochs):
        # Randomly select batches for this epoch
        batch_indices = random.sample(range(num_batches), 1)
        for idx in batch_indices:
            optimizer.zero_grad()
            start_idx = idx * batch_size
            end_idx = (idx + 1) * batch_size
            batch_x, batch_y = X_train[start_idx:end_idx], Y_train[start_idx:end_idx]
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            if loss < min_loss:
                min_loss = loss
                best_model_state = model.state_dict()  # Save best model state

        print(
            f'\rEpoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}, Min Loss: {min_loss}, LR = {optimizer.param_groups[0]["lr"]:.7f}',
            end="")

        if min_loss < early_stopping_threshold:
            epochs = epoch
            break

    print()
    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        # Restore best model state for evaluation
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        train_loss = min_loss
        outputs_test = model(X_test)
        test_loss = criterion(outputs_test, Y_test).item()

        for i in range(10):
            print(f'Prediction: {outputs_test[i]}, Actual: {Y_test[i]}')

    results = {
        'Epochs': epochs,
        'Layer Sizes': layer_sizes,
        'Learning Rate': learning_rate,
        'Weight Decay': weight_decay,
        'Train Loss': train_loss.item(),
        'Test Loss': test_loss,
        'Number of Layers': len(layer_sizes),
        'Input Dim': input_dim
    }

    return min_loss, model, test_loss, epochs


# Load the layers from the original CSV file
with open('pq_search_results/100_layers.csv', mode='r') as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        input_dimension = int(row['Input dimension'])
        num_layers = int(row['Number of Layers'])
        factor_q = float(row['Factor q'])
        meant_complexity = int(row['Meant Complexity'])
        actual_complexity = int(row['Actual Complexity'])
        first_layer = int(row['First Layer'])
        layers = list(map(int, row['Layers'].strip('"').split(',')))

        for i in range(3):
            dataset_size = 10000
            print("Dataset size:", dataset_size)
            X, Y = generate_dataset(dataset_size, input_dimension)

            # Split the data into training and testing sets
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05, random_state=0)

            # Convert to PyTorch tensors
            X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
            Y_train = torch.tensor(Y_train, dtype=torch.float32).view(-1, 1).to(device)
            X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
            Y_test = torch.tensor(Y_test, dtype=torch.float32).view(-1, 1).to(device)

            # Train the model with the loaded layer sizes
            learning_rate = 0.01  # Replace with desired value
            weight_decay = 0.0001  # Replace with desired value
            loss_tolerance = 0.01
            print(f'Training with layers {layers}, lr {learning_rate}, wd {weight_decay}')
            train_loss, model, test_loss, epochs = train_and_evaluate_model(input_dimension, layers, learning_rate, weight_decay, early_stopping_threshold=loss_tolerance)

            del X_train, X_test, Y_train, Y_test

            # Save the results alongside the original data into a new CSV
            new_row = {
                'Input dimension': input_dimension,
                'Number of Layers': num_layers,
                'Factor q': factor_q,
                'Meant Complexity': meant_complexity,
                'Actual Complexity': actual_complexity,
                'First Layer': first_layer,
                'Layers': row['Layers'],
                'Dataset size': dataset_size,
                'Train Loss': train_loss.item(),
                'Test Loss': test_loss,
                'Epochs': epochs
            }

            with open('data_demand_and_complexity_results/results_100_layers.csv', mode='a', newline="") as result_file:
                fieldnames = ['Input dimension', 'Number of Layers', 'Factor q', 'Meant Complexity', 'Actual Complexity',
                              'First Layer', 'Layers', 'Dataset size', 'Train Loss', 'Test Loss', 'Epochs']
                writer = csv.DictWriter(result_file, fieldnames=fieldnames)

                if result_file.tell() == 0:
                    writer.writeheader()  # Write header only once

                writer.writerow(new_row)

            if test_loss < 0.1:
                break
