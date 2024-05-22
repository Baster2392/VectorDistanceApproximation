import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
from training_and_searching.sample_based_distance.eb_training import *
from models.siamese_model import SiameseNetwork

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix

from sklearn.linear_model import LogisticRegression

CSV_FILE_PATH = '../../saved_results/res2.csv'


if __name__ == '__main__':
    dataset = []


    df = pd.read_csv('../databases/penguins_lter.csv', skiprows=1, header=None)

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



    # Extract class labels
    class_labels = [row[0] for row in dataset]

    original_class_counts = {label: class_labels.count(label) for label in set(class_labels)}


    # Divide dataset into training and test data with stratification
    X_train, X_test, y_train, y_test = train_test_split(data_list, class_labels, test_size=0.9,
                                                        random_state=42, stratify=class_labels)
    X_train = [sublist[1:] for sublist in X_train]
    X_test = [sublist[1:] for sublist in X_test]

    train_class_counts = {label: y_train.count(label) for label in set(class_labels)}

    print("Original Class Counts:")
    print(original_class_counts)
    print("\nX_train Class Counts:")
    print(train_class_counts)

    species = [
        ("Adelie Penguin (Pygoscelis adeliae)", "Chinstrap penguin (Pygoscelis antarctica)", 10),
        ("Adelie Penguin (Pygoscelis adeliae)", "Gentoo penguin (Pygoscelis papua)", 20),
        ("Chinstrap penguin (Pygoscelis antarctica)", "Gentoo penguin (Pygoscelis papua)", 50)
    ]

    similarity_database = build_similarity_database(X_train, y_train, species)

    #traing the model with the similarity database/ test dataset

    avg_distance = calculate_average_distance(similarity_database)
    print("Average Distance in Similarity Database:", avg_distance)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SiameseNetwork(6, 90, 8 ,3)
    criterion = nn.L1Loss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=500, factor=0.5, min_lr=1e-8, verbose=True)

    model, _, _, _ = train(model, custom_loss, optimizer, scheduler, epochs=100000, n_samples=50,
                           similarity_database=similarity_database, loss_tolerance=2, device=device)

    # Update similarity database using the trained model
 #   similarity_database = update_similarity_database(model, X_train, dataset, similarity_database)

    print(f'=====================================================')
    print(f'=====================================================')


    # Train the model
 #   model, _, _, _ = train(model, custom_loss, optimizer, scheduler, epochs=100000, n_samples=200,
  #                         similarity_database=similarity_database, loss_tolerance=1, device=device)



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

    classifications_euclidean = classify_samples_euclidean(model, centroids, X_test)
    accuracy_euclidean = calculate_accuracy(classifications_euclidean, y_test)
    print(f'Accuracy with Euclidean distance: {accuracy_euclidean:.2f}%')

    # Preprocess the data
    X_train_logistic = np.array(X_train)
    X_test_logistic = np.array(X_test)
    y_train_logistic = np.array([label.split(",")[0].strip() for label in y_train])
    y_test_logistic = np.array([label.split(",")[0].strip() for label in y_test])

    # Initialize the Logistic Regression model
    logistic_model = LogisticRegression(max_iter=10000)

    # Train the model on the training data
    logistic_model.fit(X_train_logistic, y_train_logistic)

    # Predict the labels of the test data
    y_pred_logistic = logistic_model.predict(X_test_logistic)

    # Evaluate the performance of the model
    accuracy = accuracy_score(y_test_logistic, y_pred_logistic)
    f1 = f1_score(y_test_logistic, y_pred_logistic, average='weighted')
    recall = recall_score(y_test_logistic, y_pred_logistic, average='weighted')
    precision = precision_score(y_test_logistic, y_pred_logistic, average='weighted')
    conf_matrix = confusion_matrix(y_test_logistic, y_pred_logistic)

    print(f'Accuracy: {accuracy:.2f}%')
    print(f'F1 Score: {f1:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Confusion Matrix:\n{conf_matrix}')

    #Plot the classifications
    plot_classification(X_test, classifications, "Model")
    plot_classification(X_test_logistic, y_pred_logistic, "Library")
    plot_classification(X_test, classifications_euclidean, "Euclidean")


