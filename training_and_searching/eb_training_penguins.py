import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from eb_training import calculate_accuracy
from eb_training import classify_samples
from eb_training import generate_sample_data
from eb_training import validate
from eb_training import custom_loss
from eb_training import train

from models.siamese_model import SiameseNetwork
from data_generators import vector_generator as vg

CSV_FILE_PATH = '../saved_results/res2.csv'



def plot_classification(X, classifications):
    # Define colors for each class
    colors = {'Adelie Penguin (Pygoscelis adeliae)': 'r', 'Gentoo penguin (Pygoscelis papua)': 'g', 'Chinstrap penguin (Pygoscelis antarctica)': 'b'}

    # Create scatter plots for each class
    for cls in set(classifications):
        x_values = [X[i][2] for i in range(len(X)) if classifications[i] == cls and X[i][2] != 0]
        y_values = [X[i][3] for i in range(len(X)) if classifications[i] == cls and X[i][3] != 0]
        plt.scatter(x_values, y_values, color=colors[cls], label=cls)

    # Add legend
    plt.legend()

    # Add labels and title
    plt.xlabel('Flipper length')
    plt.ylabel('Body Mass')
    plt.title('Classification of penguins')

    # Show plot
    plt.show()


def initialize_centroids(X, y):
    unique_classes = np.unique([sample.split(",")[0].strip() for sample in y])  # Get unique classes
    centroids = {}

    for cls in unique_classes:
        class_samples = np.array(X)[[sample.split(",")[0].strip() == cls for sample in y]]
        centroid = np.mean(class_samples, axis=0)
        centroids[cls] = centroid
    return centroids






if __name__ == '__main__':
    dataset = []


    df = pd.read_csv('databases/penguins_lter.csv', skiprows=1, header=None)

    columns_to_ignore = [0, 1, 3, 4, 5, 6, 7, 8, 13, 16]

    df_filtered = df[df.columns.difference(df.columns[columns_to_ignore])]

    df_filtered.fillna(0, inplace=True)

    data_list = df_filtered.values.tolist()

    vectors = []
    for row in dataset:
        vector = [float(val) for val in row[0].split(',')[:-1]]
        vectors.append(vector)
    dataset = data_list

    vectors = [sublist[1:] for sublist in data_list]

    # Divide dataset into training and test data
    X_train, X_test, y_train, y_test = train_test_split(data_list, [row[0] for row in dataset], test_size=0.95,
                                                        random_state=42)
    X_train = [sublist[1:] for sublist in X_train]
    X_test = [sublist[1:] for sublist in X_test]


    # Define Similarity Database
    similarity_database = {}
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            flower1 = dataset[i][0]
            flower2 = dataset[j][0]
            similarity = 0
            if flower1 == flower2:
                similarity = 0
            elif (flower1 == "Adelie Penguin (Pygoscelis adeliae)" and flower2 == "Chinstrap penguin (Pygoscelis antarctica)") or \
                    (flower1 == "Chinstrap penguin (Pygoscelis antarctica)" and flower2 == "Adelie Penguin (Pygoscelis adeliae)"):
                similarity = 5
            elif (flower1 == "Adelie Penguin (Pygoscelis adeliae)" and flower2 == "Gentoo penguin (Pygoscelis papua)") or \
                    (flower1 == "Gentoo penguin (Pygoscelis papua)" and flower2 == "Adelie Penguin (Pygoscelis adeliae)"):
                similarity = 10
            elif (flower1 == "Chinstrap penguin (Pygoscelis antarctica)" and flower2 == "Gentoo penguin (Pygoscelis papua)") or \
                    (flower1 == "Gentoo penguin (Pygoscelis papua)" and flower2 == "Chinstrap penguin (Pygoscelis antarctica)"):
                similarity = 15
            similarity_database[(tuple(vectors[i]), tuple(vectors[j]))] = similarity
            similarity_database[(tuple(vectors[j]), tuple(vectors[i]))] = similarity  # Similarity matrix is symmetric

    #traing the model with the similarity database/ test dataset

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SiameseNetwork(6, 600, 1)
    criterion = nn.L1Loss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=200, factor=0.75, min_lr=1e-8, verbose=True)

    model, _, _, _ = train(model, custom_loss, optimizer, scheduler, epochs=100000, n_samples=100,
                           similarity_database=similarity_database, loss_tolerance=1.7, device=device)

    x_validate, y_validate = generate_sample_data(200, 0, 100000, model.input_dim, similarity_database, False)
    x_validate = torch.tensor(x_validate, dtype=torch.float).to(device)
    y_validate = torch.tensor(y_validate, dtype=torch.float).to(device)

    validate(model, criterion, x_validate, y_validate)

    #Initialize centroids for each class
    centroids = initialize_centroids(X_test, y_test)

    # Classify all samples in the dataset
    classifications = classify_samples(model, centroids, X_test)
    accuracy = calculate_accuracy(classifications, y_test)
    print(f'Accuracy: {accuracy:.2f}%')

    #Plot the classifications
    plot_classification(X_test, classifications)
