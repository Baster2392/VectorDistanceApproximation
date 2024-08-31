import itertools

import matplotlib.pyplot as plt
import metric_learn as ml
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.init as init
from sklearn import datasets
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


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

        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                init.kaiming_uniform_(param.data)
            elif 'bias' in name:
                init.zeros_(param.data)
        for layer in self.fc:
            init.kaiming_uniform(layer.weight)

    def forward(self, input):
        out, _ = self.rnn(input)
        out = out[:, -1, :]
        for i, layer in enumerate(self.fc):
            out = layer(out)
            if i < len(self.fc) - 1:
                out = nn.functional.relu(out)
        return out


class LinearModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers=3):
        super(LinearModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            if i == 0:
                if self.num_layers == 1:
                    self.layers.append(nn.Linear(self.input_dim, self.output_dim))
                else:
                    self.layers.append(nn.Linear(self.input_dim, self.hidden_dim))
            elif i == self.num_layers - 1:
                self.layers.append(nn.Linear(self.hidden_dim, self.output_dim))
            else:
                self.layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = nn.functional.leaky_relu(x)
        return x


def create_dataset(alg='lmnn'):
    iris = datasets.load_iris()
    X_train = iris.data
    y_train = iris.target

    if alg == 'LMNN':
        algorythm = ml.LMNN(n_neighbors=3, learn_rate=1e-6)
    elif alg == 'NCA':
        algorythm = ml.NCA()
    elif alg == 'RCA':
        algorythm = ml.RCA()

    algorythm.fit(X_train, y_train)
    X_train_transformed = algorythm.transform(X_train)

    euclidean_distances = pairwise_distances(X_train, metric='euclidean')
    covariance_matrix = np.cov(X_train_transformed, rowvar=False)
    distances_learned = pairwise_distances(X_train_transformed, metric='mahalanobis', VI=np.linalg.inv(covariance_matrix))

    pairs = []
    similarities_learned = []
    for i in range(len(X_train_transformed)):
        for j in range(i + 1, len(X_train_transformed)):
            euclidean_distance = euclidean_distances[i, j]
            learned_distance = distances_learned[i, j]
            similarity_euclidean = 1 / (1 + euclidean_distance)
            similarity_learned = 1 / (1 + learned_distance)
            similarities_learned.append(similarity_learned)
            pairs.append((i, j, euclidean_distance, similarity_euclidean, learned_distance, similarity_learned))

    return X_train, similarities_learned


def get_data_as_tensor(mode='recurrent', alg='lmnn'):   # recurrent, linear
    iris_X, iris_learned_metric = create_dataset(alg=alg)
    scaler = StandardScaler()
    iris_X = scaler.fit_transform(iris_X)
    iris_X, iris_learned_metric = torch.tensor(iris_X, dtype=torch.float), torch.tensor(iris_learned_metric, dtype=torch.float)

    batch_number = iris_X.shape[0]

    i_idx, j_idx = [], []
    for i in range(batch_number):
        for j in range(i + 1, batch_number):
            i_idx.append(i)
            j_idx.append(j)

    if mode == 'recurrent':
        pairs_iris = torch.stack((iris_X[i_idx], iris_X[j_idx]), dim=-1)
    elif mode == 'linear':
        print(i_idx)
        print(j_idx)
        pairs_iris = torch.cat((iris_X[i_idx], iris_X[j_idx]), dim=-1)
    return pairs_iris, iris_learned_metric


def train(settings):
    batch_size = 32
    x_dataset, y_dataset = get_data_as_tensor(settings['network_type'], settings['algorithm'])
    x_train, x_test, y_train, y_test = train_test_split(x_dataset, y_dataset, test_size=0.2)

    dataset_train = torch.utils.data.TensorDataset(x_train, y_train)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataset_test = torch.utils.data.TensorDataset(x_test, y_test)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

    if settings['network_type'] == 'recurrent':
        model = LSTMModel(2, 32, 256, 2, 3)
    elif settings['network_type'] == 'linear':
        model = LinearModel(8, 1, 64, 3)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss(reduction='mean')
    model.train()
    for epoch in range(settings['epochs']):
        total_loss = 0.0
        for batch, (x_batch, y_batch) in enumerate(dataloader_train):
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print("Epoch {}, Loss {}".format(epoch, total_loss))

    model.eval()
    with torch.no_grad():
        y_pred = model(x_test)
        errors = y_test - y_pred.squeeze()
    average_absolute_loss = torch.mean(torch.abs(errors))
    plot_errors(errors, average_absolute_loss, settings)


def plot_errors(error_tensor, average_absolute_loss, settings):
    error_tensor = error_tensor.cpu().detach().numpy()

    sns.histplot(error_tensor, kde=True, bins=100, stat='percent')
    plt.title(f'Error Distribution for {settings['algorithm']} algorithm after {settings['epochs']} epochs')
    plt.xlabel('Error value')
    plt.ylabel('Percentage')
    plt.text(0.55, 0.95, f'Average absolute loss: {average_absolute_loss:.4f}', transform=plt.gca().transAxes)
    plt.savefig(settings['save_path'] + f'error_distribution_{settings['epochs']}_epochs.png')
    plt.show()


if __name__ == '__main__':
    epochs_values = [20, 40, 60, 80]
    algorithms = ['LMNN', 'NCA', 'RCA']
    for algorithm, epochs in itertools.product(algorithms, epochs_values):
        settings = {
            'network_type': 'recurrent',
            'epochs': epochs,
            'algorithm': algorithm,
            'save_path': f'./saved_results/recurrent/{algorithm}/',
        }
        print("Training for settings:", settings)
        train(settings)
