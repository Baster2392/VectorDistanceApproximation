import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


"""
The code is chaotic, yet it accomplishes the task of training a Siamese Network on the penguins dataset.
However I would advise against running it - since the training batches are so large the process is very time consuming.
It could be better integrated into the rest of the project, as of right now there are some redundancies.
Particularly the Siamese Network could and frankly should be integrated into the existing model, since it is
by and large a copy of it - but this way I didn't have to enable and disable the scaling factors and so on
while experimenting. Since the accuracy is quite high, I would say that we don't need to go on trying to
make sample based distance recurrent network - the Siamese Network is enough. Of course I you have too much
time and computational power on your hands, feel free to do so.


In other words - best to leave it to me.
"""


def prepare_pairs(X, y):
    pairs = []
    labels = []
    n = len(y)
    for i in range(n):
        for j in range(i+1, n):
            pairs.append((X[i], X[j]))
            norm= np.linalg.norm(X[i]-X[j])
            norm = norm/500
            b = random.uniform(-5, 5)
            if y[i] == y[j]:
                labels.append((1+norm)/2+b)
            elif {y[i], y[j]} == {0, 2}:
                labels.append((10+norm)/2+b)
            elif {y[i], y[j]} == {0, 1}:
                labels.append((20+norm)/2+b)
            elif {y[i], y[j]} == {1, 2}:
                labels.append((50+norm)/2+b)
    return np.array(pairs), np.array(labels)


class SiameseNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_siamese_layers, num_shared_layers):
        super(SiameseNetwork, self).__init__()
        self.siamese_layers = nn.ModuleList(
            [nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim) for i in range(num_siamese_layers)]
        )
        shared_layers = []
        for _ in range(num_shared_layers):
            shared_layers.append(nn.Linear(2 * hidden_dim, 2 * hidden_dim))
            shared_layers.append(nn.ReLU())
        shared_layers.append(nn.Linear(2 * hidden_dim, 1))
        self.shared_layers = nn.Sequential(*shared_layers)

    def forward(self, x):
        x1, x2 = x[:, 0, :], x[:, 1, :]
        for layer in self.siamese_layers:
            x1 = F.relu(layer(x1))
            x2 = F.relu(layer(x2))
        concatenated = torch.cat((x1, x2), dim=1)
        output = self.shared_layers(concatenated)
        return output


def compute_centroids(X, y):
    unique_classes = np.unique(y)
    centroids = np.zeros((len(unique_classes), X.shape[1]))
    for idx, cls in enumerate(unique_classes):
        centroids[idx] = np.mean(X[y == cls], axis=0)
    return centroids

def compute_distances(model, X, centroids_tensor):
    distances = []
    X_tensor = torch.tensor(X, dtype=torch.float32)
    for i in range(X_tensor.shape[0]):
        sample = X_tensor[i].unsqueeze(0).repeat(centroids_tensor.shape[0], 1)
        pairs = torch.stack((sample, centroids_tensor), dim=1)
        with torch.no_grad():
            output = model(pairs)
        distances.append(output.view(-1).numpy())
    return np.array(distances)

def main():

    # Load the data
    data = pd.read_csv('penguins_lter.csv')
    # Drop rows with missing values
    data = data.dropna(subset=['Culmen Length (mm)', 'Culmen Depth (mm)', 'Flipper Length (mm)', 'Body Mass (g)', 'Sex'])

    # Coverting the data to numpy arrays
    X = data[['Culmen Length (mm)', 'Culmen Depth (mm)', 'Flipper Length (mm)', 'Body Mass (g)']].values
    y = data['Species'].values

    # Map the species strings to numbers
    species_map = {
        'Adelie Penguin (Pygoscelis adeliae)': 0,
        'Gentoo penguin (Pygoscelis papua)': 1,
        'Chinstrap penguin (Pygoscelis antarctica)': 2
    }
    y = np.array([species_map[species] for species in y])

    # Perform the train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Generate the combinations
    pairs_train, labels_train = prepare_pairs(X_train, y_train)

    pairs_train_tensor = torch.tensor(pairs_train, dtype=torch.float32)
    labels_train_tensor = torch.tensor(labels_train, dtype=torch.float32).view(-1, 1)

    train_dataset = TensorDataset(pairs_train_tensor, labels_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Model hiperparameters
    input_dim = X_train.shape[1]
    hidden_dim = 128
    num_siamese_layers = 3
    num_shared_layers = 2


    model = SiameseNetwork(input_dim, hidden_dim, num_siamese_layers, num_shared_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Trening
    epochs = 50
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            x_batch, y_batch = batch
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

    # Classifications


    centroids = compute_centroids(X_train, y_train)
    centroids_tensor = torch.tensor(centroids, dtype=torch.float32)



    distances = compute_distances(model, X_test, centroids_tensor)

    predicted_labels = np.argmin(distances, axis=1)

    reverse_species_map = {v: k for k, v in species_map.items()}
    y_test_labels = [reverse_species_map[label] for label in y_test]
    predicted_labels_species = [reverse_species_map[label] for label in predicted_labels]

    # Accuracy score and analysis.
    accuracy = accuracy_score(y_test, predicted_labels)
    print(f"Accuracy: {accuracy:.4f}")

    print(classification_report(y_test_labels, predicted_labels_species))


    culmen_length_test = X_test[:, 0]
    body_mass_test = X_test[:, 3]

    plt.figure(figsize=(10, 6))
    for i, species in reverse_species_map.items():
        indices = np.where(predicted_labels == i)
        plt.scatter(culmen_length_test[indices], body_mass_test[indices], label=species, alpha=0.6)

    plt.xlabel('Culmen Length (mm)')
    plt.ylabel('Body Mass (g)')
    plt.title('Culmen Length vs Body Mass (Test Data)')
    plt.legend()
    plt.show()


    # Using the shorter, common names for the penguins for higher readability.
    short_labels = {0: 'Adelie', 1: 'Gentoo', 2: 'Chinstrap'}

    conf_matrix = confusion_matrix(y_test, predicted_labels)

    conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))

    ax = sns.heatmap(conf_matrix_normalized, annot=True, fmt='.2f', cmap='Blues',
                     xticklabels=short_labels.values(), yticklabels=short_labels.values(),
                     linewidths=1, linecolor='black', cbar=True)

    plt.xticks(rotation=0)
    plt.yticks(rotation=0)

    plt.xlabel('Predicted label', fontsize=12, labelpad=10)
    plt.ylabel('True label', fontsize=12, labelpad=10)

    plt.title('Confusion Matrix', fontsize=14)

    plt.tight_layout()

    plt.show()

if __name__ == '__main__':
    main()