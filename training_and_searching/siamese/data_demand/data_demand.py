import math
import numpy as np
import torch
import torch.nn as nn
import time
import csv


class SharedNetwork(nn.Module):
    def __init__(self, num_of_inputs, layers_config):
        super(SharedNetwork, self).__init__()
        layers = []

        input_dim = num_of_inputs
        for num_neurons in layers_config[:-1]:
            layers.append(nn.Linear(input_dim, num_neurons))
            layers.append(nn.ReLU())
            input_dim = num_neurons

        layers.append(nn.Linear(input_dim, layers_config[-1]))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class SiameseNetwork(nn.Module):
    def __init__(self, input_dim, layers_config):
        super(SiameseNetwork, self).__init__()
        self.shared_network = SharedNetwork(input_dim, layers_config)
        embedding_size = layers_config[-1]
        self.linear_layer_1 = nn.Linear(embedding_size, embedding_size)
        self.linear_layer_2 = nn.Linear(embedding_size, 1)

    def forward(self, input1, input2):
        embedding1 = self.shared_network(input1)
        embedding2 = self.shared_network(input2)

        difference = embedding1 - embedding2
        hidden = nn.functional.relu(self.linear_layer_1(difference))
        distance = self.linear_layer_2(hidden)
        return distance


def generate_dataset(dataset_size: int, input_dim: int):
    X1 = np.random.uniform(-1, 1, (dataset_size, input_dim))
    X2 = np.random.uniform(-1, 1, (dataset_size, input_dim))
    Y = np.sqrt(np.sum((X2 - X1) ** 2, axis=1))
    X = np.hstack((X1, X2))
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32).unsqueeze(1)


def train(train_params):
    input_dim = train_params['input_dim']
    layers_config = train_params['layers_config']
    learning_rate = train_params['learning_rate']
    criterion = train_params['criterion']
    weight_decay = train_params['weight_decay']
    max_num_epochs = train_params['max_num_epochs']
    batch_size = train_params['batch_size']
    train_dataset_size = train_params['train_dataset_size']
    val_dataset_size = train_params['val_dataset_size']
    device = train_params['device']

    train_X, train_Y = generate_dataset(train_dataset_size, input_dim)
    val_X, val_Y = generate_dataset(val_dataset_size, input_dim)
    train_X = train_X.to(device)
    train_Y = train_Y.to(device)
    val_X = val_X.to(device)
    val_Y = val_Y.to(device)

    model = SiameseNetwork(input_dim, layers_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 50

    start_time = time.time()

    for epoch in range(max_num_epochs):
        model.train()
        optimizer.zero_grad()

        indices = torch.randint(0, train_dataset_size, (batch_size,), device=device)
        X_batch = train_X[indices]
        Y_batch = train_Y[indices]

        output = model(X_batch[:, :input_dim], X_batch[:, input_dim:])
        loss = criterion(output, Y_batch)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_output = model(val_X[:, :input_dim], val_X[:, input_dim:])
            val_loss = criterion(val_output, val_Y)

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 100 == 0:
            print(f"\rEpoch: {epoch}, Loss: {loss.item()}, Val loss: {val_loss.item()}", end="")

        if epochs_no_improve >= patience:
            print("\nEarly stopping triggered")
            break

    training_time = time.time() - start_time
    print(f"\nTraining time: {training_time:.2f} seconds")
    return model, training_time, epoch + 1  # Return actual number of epochs


def evaluate_model(model, test_dataset_size, input_dim, criterion, device):
    test_X, test_Y = generate_dataset(test_dataset_size, input_dim)
    test_X = test_X.to(device)
    test_Y = test_Y.to(device)

    model.eval()
    with torch.no_grad():
        test_output = model(test_X[:, :input_dim], test_X[:, input_dim:])
        loss = criterion(test_output, test_Y)

    return loss.item()


# Training Parameters
train_params = {
    'input_dim': 25,
    'layers_config': [256, 512, 128, 64],
    'learning_rate': 0.001,
    'criterion': nn.L1Loss(),
    'weight_decay': 0.000,
    'max_num_epochs': 1000,
    'batch_size': 64,
    'train_dataset_size': 25000,
    'val_dataset_size': 100,
    'test_dataset_size': 100,
    'early_stopping_threshold': 0.1,
    'device': 'cpu'
}

# Training and Evaluation
trained_model, training_time, actual_epochs = train(train_params)
test_loss = evaluate_model(trained_model, train_params['test_dataset_size'], train_params['input_dim'],
                           train_params['criterion'], train_params['device'])
print(f"\nTest loss: {test_loss}")

# Compute Model Metrics
total_params = sum(p.numel() for p in trained_model.parameters())
num_layers = sum(1 for m in trained_model.modules() if isinstance(m, nn.Linear))

# Compute Theoretical Computational Complexity
input_dim = train_params['input_dim']
layers_config = train_params['layers_config']
shared_complexity = 0
in_dim = input_dim
for num_neurons in layers_config[:-1]:
    shared_complexity += in_dim * num_neurons
    in_dim = num_neurons
shared_complexity += in_dim * layers_config[-1]
theoretical_complexity = 2 * shared_complexity + layers_config[-1] * layers_config[-1] + layers_config[-1] * 1

# Save to CSV
csv_filename = "training_results.csv"
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
        "Test Loss", "Theoretical Complexity", "Number of Layers", "Number of Parameters",
        "Training Time (s)", "Input Dimensionality", "Batch Size", "Max Epochs",
        "Actual Epochs", "Learning Rate", "Weight Decay", "Train Dataset Size",
        "Validation Dataset Size", "Test Dataset Size"
    ])
    writer.writerow([
        test_loss, theoretical_complexity, num_layers, total_params, training_time,
        train_params['input_dim'], train_params['batch_size'], train_params['max_num_epochs'],
        actual_epochs, train_params['learning_rate'], train_params['weight_decay'],
        train_params['train_dataset_size'], train_params['val_dataset_size'],
        train_params['test_dataset_size']
    ])

print(f"Results saved to {csv_filename}")
