import torch
import torch.nn as nn
import torch.optim as optim
import csv
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from eb_training import calculate_accuracy
from eb_training import classify_samples
from eb_training import generate_sample_data
from eb_training import validate
from eb_training import custom_loss
from eb_training import train

from models.siamese_model_no_norm import SiameseNetworkNoNorm

CSV_FILE_PATH = '../saved_results/res2.csv'

def plot_classification(X, classifications):
    # Define colors for each class
    colors = {'Iris-setosa': 'r', 'Iris-versicolor': 'g', 'Iris-virginica': 'b'}

    # Create scatter plots for each class
    for cls in set(classifications):
        x_values = [X[i][0] for i in range(len(X)) if classifications[i] == cls]
        y_values = [X[i][1] for i in range(len(X)) if classifications[i] == cls]
        plt.scatter(x_values, y_values, color=colors[cls], label=cls)

    # Add legend
    plt.legend()

    # Add labels and title
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Classification of Iris Dataset')

    # Show plot
    plt.show()



def initialize_centroids(X, y):
    unique_classes = np.unique([sample.split(",")[-1].strip() for sample in y])
    centroids = {}
    for cls in unique_classes:
        class_samples = np.array(X)[[sample.split(",")[-1].strip() == cls for sample in y]]
        centroid = np.mean(class_samples, axis=0)
        centroids[cls] = centroid
    return centroids





if __name__ == '__main__':
    dataset = []
    with open('databases/iris_dataset.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        next(reader)  # Skip the header
        for row in reader:
            dataset.append(row)


    # Create vectors for each flower sample
    vectors = []
    for row in dataset:
        # Split the row by comma and convert the first four values to floats
        vector = [float(val) for val in row[0].split(',')[:-1]]
        vectors.append(vector)

    # Define Similarity Database
    similarity_database = {}
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            flower1 = dataset[i][-1].split(",")[-1].strip()  # Get the species label for the first flower
            flower2 = dataset[j][-1].split(",")[-1].strip()  # Get the species label for the second flower
            similarity = 0  # Default similarity score
            if flower1 == flower2:
                similarity = 0
            elif (flower1 == "Iris-versicolor" and flower2 == "Iris-virginica") or \
                    (flower1 == "Iris-virginica" and flower2 == "Iris-versicolor"):
                similarity = 5
            elif (flower1 == "Iris-versicolor" and flower2 == "Iris-setosa") or \
                    (flower1 == "Iris-setosa" and flower2 == "Iris-versicolor"):
                similarity = 10
            elif (flower1 == "Iris-virginica" and flower2 == "Iris-setosa") or \
                    (flower1 == "Iris-setosa" and flower2 == "Iris-virginica"):
                similarity = 15
            similarity_database[(tuple(vectors[i]), tuple(vectors[j]))] = similarity
            similarity_database[(tuple(vectors[j]), tuple(vectors[i]))] = similarity  # Similarity matrix is symmetric

    # Divide dataset into training and test data
    X_train, X_test, y_train, y_test = train_test_split(vectors, [row[-1] for row in dataset], test_size=0.2,
                                                        random_state=42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SiameseNetworkNoNorm(4, 400, 1)
    criterion = nn.L1Loss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=200, factor=0.75, min_lr=1e-8, verbose=True)

    model, _, _, _ = train(model, custom_loss, optimizer, scheduler, epochs=100000, n_samples=100,
                           similarity_database=similarity_database, loss_tolerance=1, device=device)

    x_validate, y_validate = generate_sample_data(200, 0, 100000, model.input_dim, similarity_database, False)
    x_validate = torch.tensor(x_validate, dtype=torch.float).to(device)
    y_validate = torch.tensor(y_validate, dtype=torch.float).to(device)

    validate(model, criterion, x_validate, y_validate)

    # Initialize centroids for each class using the entire dataset
    centroids = initialize_centroids(vectors, [row[-1] for row in dataset])

    # Classify all samples in the dataset
    classifications = classify_samples(model, centroids, vectors)
    accuracy = calculate_accuracy(classifications, [sample[-1].split(",")[-1].strip() for sample in dataset])
    print(f'Accuracy: {accuracy:.2f}%')

    # Plot the classifications
    plot_classification(vectors, classifications)
