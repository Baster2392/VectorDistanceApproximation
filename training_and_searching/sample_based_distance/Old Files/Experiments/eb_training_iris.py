import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import pandas as pd
from training_and_searching.sample_based_distance.eb_training import *
from models.siamese_model import SiameseNetwork

CSV_FILE_PATH = '../../../../saved_results/res2.csv'






if __name__ == '__main__':
    dataset = []

    dataset = []

    df = pd.read_csv('../databases/iris_dataset.csv', skiprows=1, header=None)

    for row in df.values:
        features = row[1:].astype(float)  # Extract features from the row, converting them to float
        dataset.append(features)

    class_labels = df[0].tolist()


    original_class_counts = {label: class_labels.count(label) for label in set(class_labels)}

    # Divide dataset into training and test data with stratification
    X_train, X_test, y_train, y_test = train_test_split(dataset, class_labels, test_size=0.8,
                                                        random_state=42, stratify=class_labels)

    train_class_counts = {label: y_train.count(label) for label in set(class_labels)}

    print("Original Class Counts:")
    print(original_class_counts)
    print("\nX_train Class Counts:")
    print(train_class_counts)

    # Define Similarity Database
    species = [
        ("Iris-setosa", "Iris-virginica", 50),
        ("Iris-setosa", "Iris-versicolor", 20),
        ("Iris-virginica", "Iris-versicolor", 10)
    ]

    similarity_database = build_similarity_database(X_train, y_train, species)
    # Training the model with the similarity database/test dataset

    avg_distance = calculate_average_distance(similarity_database)
    print("Average Distance in Similarity Database:", avg_distance)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SiameseNetwork(4, 80, 3)  # Change input dimension to match iris dataset (4 features)
    criterion = nn.L1Loss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=500, factor=0.5, min_lr=1e-8, verbose=True)

    model, _, _, _ = train(model, custom_loss, optimizer, scheduler, epochs=100000, n_samples=50,
                           similarity_database=similarity_database, loss_tolerance=3, device=device)

    print(f'=====================================================')
    print(f'=====================================================')

    x_validate, y_validate = generate_sample_data(200, 0, 100000, model.input_dim, similarity_database, False)
    x_validate = torch.tensor(x_validate, dtype=torch.float).to(device)
    y_validate = torch.tensor(y_validate, dtype=torch.float).to(device)

    validate(model, criterion, x_validate, y_validate)

    # Initialize centroids for each class
    centroids = initialize_centroids(X_test, y_test)

    # Classify all samples in the dataset
    classifications = classify_samples(model, centroids, X_test)
    accuracy = calculate_accuracy(classifications, y_test)
    print(f'Accuracy: {accuracy:.2f}%')

    classifications_euclidean = classify_samples_euclidean(model, centroids, X_test)
    accuracy_euclidean = calculate_accuracy(classifications_euclidean, y_test)
    print(f'Accuracy with Euclidean distance: {accuracy_euclidean:.2f}%')

    # Plot the classifications
    plot_classification(X_test, classifications, "iris")
