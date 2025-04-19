import math
import numpy as np
import torch
import torch.nn as nn
import time
import csv
import itertools

# Nowa architektura transformera do obliczania dystansu między parą wektorów
class TransformerDistance(nn.Module):
    def __init__(self, input_dim, layers_config):
        """
        Parametry:
         - input_dim: wymiar wejściowych wektorów
         - layers_config: lista hiperparametrów; 
             layers_config[0] -> d_model (wymiar przestrzeni osadzeń)
             layers_config[1] -> dim_feedforward (rozmiar warstwy feedforward); opcjonalnie
        """
        super(TransformerDistance, self).__init__()
        # Ustawienia domyślne
        d_model = layers_config[0] if len(layers_config) > 0 else 128
        dim_feedforward = layers_config[1] if len(layers_config) > 1 else 256
        num_heads = 4
        num_layers = 2
        dropout = 0.1
        
        # Rzutowanie każdego z wektorów do przestrzeni d_model
        self.token_embedding = nn.Linear(input_dim, d_model)
        
        # Transformer Encoder (batch_first=True dla danych o kształcie [batch, seq_len, d_model])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=num_heads, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Warstwa regresyjna, która po pooling’u daje pojedynczą wartość – przewidywany dystans
        self.regressor = nn.Linear(d_model, 1)

    def forward(self, input1, input2):
        # input1, input2: [batch_size, input_dim]
        # Składamy je w jeden tensor o kształcie [batch_size, 2, input_dim]
        x = torch.stack((input1, input2), dim=1)
        # Rzutowanie do przestrzeni d_model
        x = self.token_embedding(x)  # [batch_size, 2, d_model]
        # Przetwarzanie przez enkoder transformera
        x = self.transformer(x)        # [batch_size, 2, d_model]
        # Pooling – średnia z obu tokenów
        x = x.mean(dim=1)              # [batch_size, d_model]
        # Regresja do pojedynczej wartości
        out = self.regressor(x)        # [batch_size, 1]
        return torch.relu(out)         # Zapewnienie, że dystans jest nieujemny

def generate_dataset(dataset_size: int, input_dim: int):
    # Generujemy dane wejściowe z rozkładu jednostajnego na [0, 1]
    X1 = np.random.uniform(0, 1, (dataset_size, input_dim))
    X2 = np.random.uniform(0, 1, (dataset_size, input_dim))
    # Obliczamy oryginalny dystans euklidesowy między wektorami (target)
    Y = np.sqrt(np.sum((X2 - X1) ** 2, axis=1))
    
    # Dodajemy normalizację danych wejściowych (standaryzacja)
    # Dla rozkładu jednostajnego [0,1] średnia wynosi 0.5, a odchylenie standardowe ~0.288675 (1/sqrt(12))
    sigma = 1 / np.sqrt(12)
    X1_norm = (X1 - 0.5) / sigma
    X2_norm = (X2 - 0.5) / sigma
    
    # Łączymy znormalizowane dane w jeden tensor [dataset_size, 2 * input_dim]
    X = np.hstack((X1_norm, X2_norm))
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

    # Generowanie zbiorów treningowego i walidacyjnego
    train_X, train_Y = generate_dataset(train_dataset_size, input_dim)
    val_X, val_Y = generate_dataset(val_dataset_size, input_dim)
    train_X = train_X.to(device)
    train_Y = train_Y.to(device)
    val_X = val_X.to(device)
    val_Y = val_Y.to(device)

    # Inicjalizacja modelu transformera
    model = TransformerDistance(input_dim, layers_config).to(device)
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
        # Dla transformera dzielimy dane na dwa wejścia: pierwsze i drugie n-wymiarowe wektory
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
    return model, training_time, epoch + 1  # Zwracamy model, czas treningu i liczbę epok

def evaluate_model(model, test_dataset_size, input_dim, criterion, device):
    test_X, test_Y = generate_dataset(test_dataset_size, input_dim)
    test_X = test_X.to(device)
    test_Y = test_Y.to(device)

    model.eval()
    with torch.no_grad():
        test_output = model(test_X[:, :input_dim], test_X[:, input_dim:])
        loss = criterion(test_output, test_Y)

    return loss.item()

def calculate_complexity(train_params):
    layers_config = train_params['layers_config']
    # Compute Theoretical Computational Complexity (uproszczony szacunek)
    input_dim_value = train_params['input_dim']
    d_model = layers_config[0]          # wymiar osadzenia
    dim_feedforward = layers_config[1] if len(layers_config) > 1 else 256
    seq_len = 2                         # mamy dwa wektory jako tokeny
    num_transformer_layers = 2          # zgodnie z implementacją w TransformerDistance

    complexity_embedding = 2 * (input_dim_value * d_model)
    complexity_attention = num_transformer_layers * (seq_len ** 2 * d_model)
    complexity_feedforward = num_transformer_layers * (seq_len * d_model * dim_feedforward)
    complexity_regression = d_model

    return complexity_embedding + complexity_attention + complexity_feedforward + complexity_regression

def summarise_training(train_params, trained_model, training_time, actual_epochs):
    test_loss = evaluate_model(trained_model, train_params['test_dataset_size'], train_params['input_dim'],
                                 train_params['criterion'], train_params['device'])
    print(f"\nTest loss: {test_loss}")

    theoretical_complexity = calculate_complexity(train_params)

    # Zapis do CSV
    csv_filename = "D:\\Studia\\Sem 4\\SI\\Projekt\\VectorDistanceCalculator\\training_and_searching\\transformer\\data_demand\\results\\data_demand1.csv"
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            test_loss, theoretical_complexity, train_params['layers_config'][0], "total_params_placeholder", training_time,
            train_params['input_dim'], train_params['batch_size'], train_params['max_num_epochs'],
            actual_epochs, train_params['learning_rate'], train_params['weight_decay'],
            train_params['train_dataset_size'], train_params['val_dataset_size'],
            train_params['test_dataset_size']
        ])

    print(f"Results saved to {csv_filename}")

# Ustawienie hiperparametrów
input_dim = [200]
i = 31
# W layers_config wykorzystujemy pierwsze dwie wartości jako d_model i dim_feedforward
layers_config = [[128 * i, 98 * i]]  
learning_rate = [0.001]
criterion = [nn.MSELoss()]
weight_decay = [0.001]
max_num_epochs = [1000000]
batch_size = [64]
train_dataset_size = [2000 + 2000 * i for i in range(0, 5)]
val_dataset_size = [100]
test_dataset_size = [10000]
early_stopping_threshold = [0.01]
device = ['cuda' if torch.cuda.is_available() else 'cpu']

# Generacja kombinacji hiperparametrów
param_combinations = list(itertools.product(
    input_dim, layers_config, learning_rate, criterion, weight_decay, max_num_epochs, batch_size, 
    train_dataset_size, val_dataset_size, test_dataset_size, early_stopping_threshold, device
))

# Trening i ewaluacja dla każdej kombinacji
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

    #print(f"Complexity: {calculate_complexity(train_params)}")
    #exit(0)
    print(f"Training with parameters: {train_params}")
    trained_model, training_time, actual_epochs = train(train_params)
    summarise_training(train_params, trained_model, training_time, actual_epochs)
