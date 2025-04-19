import math
import numpy as np
import torch
import torch.nn as nn
import time
import csv
import itertools


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
        self.input_dim = input_dim
        self.shared_network = SharedNetwork(input_dim, layers_config)
        embedding_size = layers_config[-1]
        self.linear_layer_1 = nn.Linear(embedding_size, embedding_size)
        self.linear_layer_2 = nn.Linear(embedding_size, 1)

    def forward(self, input1, input2):
        scaling_factor = math.sqrt(self.input_dim)
        embedding1 = self.shared_network(input1)
        embedding2 = self.shared_network(input2)

        difference = embedding1 - embedding2
        hidden = nn.functional.relu(self.linear_layer_1(difference))
        distance = self.linear_layer_2(hidden)
        return distance / scaling_factor


def generate_dataset(dataset_size: int, input_dim: int):
    X1 = np.random.uniform(0, 1, (dataset_size, input_dim))
    X2 = np.random.uniform(0, 1, (dataset_size, input_dim))
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
    early_stopping_threshold = train_params['early_stopping_threshold']
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
    optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_train_loss = float('inf')
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

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        if loss < best_train_loss:
            best_train_loss = loss

        if epoch % 100 == 0:
            print(f"\rEpoch: {epoch}, Loss: {loss.item()}, Val loss: {val_loss.item()}, Best train loss: {best_train_loss.item()}, Best val loss: {best_val_loss.item()}", end="")

        if loss.item() < early_stopping_threshold:
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


def summarise_training(train_params, trained_model, training_time, actual_epochs):
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
    csv_filename = "D:\\Studia\\Sem 4\\SI\\Projekt\\VectorDistanceCalculator\\training_and_searching\\siamese\\data_demand\\results\\data_demand2.csv"
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        #writer.writerow([
        #"Test Loss", "Theoretical Complexity", "Number of Layers", "Number of Parameters",
        #"Training Time (s)", "Input Dimensionality", "Batch Size", "Max Epochs",
        #"Actual Epochs", "Learning Rate", "Weight Decay", "Train Dataset Size",
        #"Validation Dataset Size", "Test Dataset Size"
    #])
        writer.writerow([
        test_loss, theoretical_complexity, num_layers, total_params, training_time,
        train_params['input_dim'], train_params['batch_size'], train_params['max_num_epochs'],
        actual_epochs, train_params['learning_rate'], train_params['weight_decay'],
        train_params['train_dataset_size'], train_params['val_dataset_size'],
        train_params['test_dataset_size']
    ])

    print(f"Results saved to {csv_filename}")


# Hyperparameters   
input_dim = [200]
i = 17
layers_config = [[128 * i, 100 * i, 66 * i, 52 * i]]
learning_rate = [0.01]
criterion = [nn.MSELoss()]
weight_decay = [0.0001]
max_num_epochs = [1000000]
batch_size = [64]
train_dataset_size = [1000 + 1000 * i for i in range(0, 10)]
val_dataset_size = [100]
test_dataset_size = [10000]
early_stopping_threshold = [0.01]
device = ['cuda' if torch.cuda.is_available() else 'cpu']

# Training Parameters Combinations
param_combinations = list(itertools.product(
    input_dim, layers_config, learning_rate, criterion, weight_decay, max_num_epochs, batch_size, 
    train_dataset_size, val_dataset_size, test_dataset_size, early_stopping_threshold, device
))

# Training and Evaluation for each combination
for params in param_combinations:
    train_params = {
        'input_dim': params[0],
        'layers_config': params[1],
        'learning_rate': params[2],
        'criterion': params[3],
        'weight_decay': params[4],
        'max_num_epochs': params[5],
        'batch_size': params[6],
        'train_dataset_size': params[7],
        'val_dataset_size': params[8],
        'test_dataset_size': params[9],
        'early_stopping_threshold': params[10],
        'device': params[11]
    }

    print(f"Training with parameters: {train_params}")
    trained_model, training_time, actual_epochs = train(train_params)
    summarise_training(train_params, trained_model, training_time, actual_epochs)
