import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

CSV_FILE_PATH = '../../../../saved_results/res2.csv'


def calculate_accuracy(classifications, y_test):
    correct = 0
    total = len(classifications)

    for predicted_class, true_class in zip(classifications, y_test):
        if predicted_class == true_class:
            correct += 1

    accuracy = (correct / total) * 100
    return accuracy

def moving_average(data, window_size):
    """Calculate the moving average of data."""
    weights = np.repeat(1.0, window_size) / window_size
    return np.convolve(data, weights, 'valid')

def classify_samples(model, centroids, X_test):
    classifications = []
    for sample in X_test:
        min_distance = float('inf')
        closest_class = None
        for cls, centroid in centroids.items():
            input_pair = np.array([sample, centroid], dtype=np.float32)
            input_pair = torch.tensor(input_pair).unsqueeze(0)
            output = model(input_pair,"custom")
            distance = output.item()
            if distance < min_distance:
                min_distance = distance
                closest_class = cls
        classifications.append(closest_class)
    return classifications


def generate_sample_data(number_of_samples, min_value, max_value, vector_size, similarity_database, for_recurrent=False):
    if not for_recurrent:
        vector_pairs = np.zeros((number_of_samples, 2, vector_size))
        similarities = np.zeros(number_of_samples)
        keys = list(similarity_database.keys())
        values = list(similarity_database.values())
        for i in range(number_of_samples):
            # Randomly select a pair of vectors from the similarity database
            index = np.random.randint(len(keys))
            vector_pair, similarity = keys[index], values[index]
            vector_pairs[i][0] = vector_pair[0]
            vector_pairs[i][1] = vector_pair[1]
            similarities[i] = similarity
    else:
        vector_pairs = np.zeros((number_of_samples, 2, vector_size, 1))
        similarities = np.zeros(number_of_samples)
        keys = list(similarity_database.keys())
        values = list(similarity_database.values())
        for i in range(number_of_samples):
            index = np.random.randint(len(keys))
            vector_pair, similarity = keys[index], values[index]
            vector_pairs[i][0] = vector_pair[0]
            vector_pairs[i][1] = vector_pair[1]
            similarities[i] = similarity
    return vector_pairs, similarities


def build_similarity_database(X_train, y_train, species):
    similarity_database = {}
    for i in range(len(X_train)):
        for j in range(i + 1, len(X_train)):
            flower1 = y_train[i]
            flower2 = y_train[j]
            similarity = 0
            # Convert lists to numpy arrays
            X_train_i = np.array(X_train[i])
            X_train_j = np.array(X_train[j])
            # Calculate Euclidean distance
            norm = (np.linalg.norm(X_train_i - X_train_j, ord=2))/200
            difference = np.max(np.abs(X_train_i - X_train_j))/200
            if flower1 == flower2:
                similarity = 1 + norm -difference
            elif (flower1 == species[0][0] and flower2 == species[0][1]) or \
                    (flower1 == species[0][1] and flower2 == species[0][0]):
                similarity = species[0][2] + norm - difference
            elif (flower1 == species[1][0] and flower2 == species[1][1]) or \
                    (flower1 == species[1][1] and flower2 == species[1][0]):
                similarity = species[1][2] + norm - difference
            elif (flower1 == species[2][0] and flower2 == species[2][1]) or \
                    (flower1 == species[2][1] and flower2 == species[2][0]):
                similarity = species[2][2] + norm - difference
            similarity_database[(tuple(X_train[i]), tuple(X_train[j]))] = similarity
    return similarity_database



def validate(model, criterion, x_validate, y_validate, window_size=25):
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        y_pred = model(x_validate, "custom")
        loss = criterion(y_pred, y_validate)

    # Sorting by actual values
    sorted_indices = torch.argsort(y_validate)
    y_validate_sorted = y_validate[sorted_indices]
    y_pred_sorted = y_pred[sorted_indices]

    # Flatten y_pred_sorted to ensure it's one-dimensional
    y_pred_sorted_flat = y_pred_sorted.cpu().numpy().flatten()

    # Plotting predicted vs actual values
    plt.figure(figsize=(8, 6))
    plt.plot(y_validate_sorted.cpu().numpy(), label='Actual', color='blue')

    plt.plot(y_pred_sorted.cpu().numpy(), label='Predicted', color='green')

    # Calculate moving average for the whole length
    padded_y_pred_sorted = np.pad(y_pred_sorted_flat, (window_size // 2, window_size // 2), mode='edge')
    moving_avg = np.convolve(padded_y_pred_sorted, np.ones(window_size)/window_size, mode='valid')

    # Adjust the x-axis to align with the moving average
    x_axis_adjusted = np.arange(len(y_pred_sorted_flat))

    plt.plot(x_axis_adjusted, moving_avg, label=f'Moving Average ({window_size})', color='orange')

    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Actual vs Predicted')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Print metrics
    print("Mean loss:", loss.item())
    print("Max loss:", torch.max(abs(y_pred_sorted - y_validate_sorted)))
    print("Min loss:", torch.min(abs(y_pred_sorted - y_validate_sorted)))
    print("Mean error:", torch.mean(abs(y_pred_sorted - y_validate_sorted) / y_validate_sorted))
    print("Min error:", torch.min(abs(y_pred_sorted - y_validate_sorted) / y_validate_sorted))
    print("Max error:", torch.max(abs(y_pred_sorted - y_validate_sorted) / y_validate_sorted))

    elapsed_time = time.time() - start_time
    print(f"Time Taken: {elapsed_time} seconds")

# Example usage
# validate(model, criterion, x_validate, y_validate, window_size=5)


def custom_loss(output, y_train, similarity_database):
    loss = 0
    for i in range(len(y_train)):
        loss += abs(output[i] - y_train[i])
    return loss / len(y_train)

def train(model, criterion, optimizer, scheduler, epochs, n_samples, similarity_database,
          loss_tolerance=0.5, device=torch.device('cpu')):
    model.to(device)

    # Training loop
    model.train()
    epoch = 0
    loss = 0
    for epoch in range(epochs):
        # Generate training data
        x_train, y_train = generate_sample_data(n_samples, 0, 1, model.input_dim, similarity_database, False)
        x_train = torch.tensor(x_train, dtype=torch.float).to(device)
        y_train = torch.tensor(y_train, dtype=torch.float).to(device)

        # Calculate loss
        optimizer.zero_grad()
        output = model(x_train, "custom")
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step(loss.item())

        # Print progress
        if epoch % 10 == 0:
            print(f'Id: {model.input_dim} Epoch [{epoch}/{epochs}], Loss: {loss.item()}, Lr: {optimizer.param_groups[0]["lr"]}')

        # Check if function converged
        if loss.item() < loss_tolerance:
            break

    return model, epoch + 1, loss.item(), optimizer.param_groups[0]["lr"]

def calculate_average_distance(similarity_database):
    distances = []
    for pair, similarity_score in similarity_database.items():
        vector_i, vector_j = pair
        distance = similarity_score
        distances.append(distance)
    average_distance = np.mean(distances)
    return average_distance

def initialize_centroids(X, y, n_clusters=1):
    unique_classes = np.unique(y)
    centroids = {}

    for cls in unique_classes:
        class_samples = np.array([X[i] for i in range(len(y)) if y[i].split(",")[0].strip() == cls])

        # Ensure data is 2D array before passing it to KMeans
        class_samples = np.atleast_2d(class_samples)

        # Use KMeans to find the centroid for the class
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans.fit(class_samples)

        # Take the first centroid (since n_clusters is 1)
        centroids[cls] = kmeans.cluster_centers_[0]

    return centroids
def classify_samples_euclidean(model, centroids, X_test):
    classifications = []
    for sample in X_test:
        min_distance = float('inf')
        closest_class = None
        for cls, centroid in centroids.items():
            # Calculate Euclidean distance between the sample and centroid
            distance = np.linalg.norm(sample - centroid)
            if distance < min_distance:
                min_distance = distance
                closest_class = cls
        classifications.append(closest_class)
    return classifications

def plot_classification(X, classifications, title):
    # Define colors and markers for each class
    class_markers = {'Iris-setosa': 'o',
                     'Iris-virginica': 'D',
                     'Iris-versicolor': '^',
                     'Adelie Penguin (Pygoscelis adeliae)': 'o',
                     'Gentoo penguin (Pygoscelis papua)': 'D',
                     'Chinstrap penguin (Pygoscelis antarctica)': '^'}

    class_colors = {'Iris-setosa': 'purple',
                    'Iris-virginica': 'c',
                    'Iris-versicolor': 'orange',
                    'Adelie Penguin (Pygoscelis adeliae)': 'purple',
                    'Gentoo penguin (Pygoscelis papua)': 'c',
                    'Chinstrap penguin (Pygoscelis antarctica)': 'orange'}

    # Create scatter plots for each class
    for cls in set(classifications):
        x_values = [X[i][0] for i in range(len(X)) if classifications[i] == cls and X[i][0] != 0]
        y_values = [X[i][1] for i in range(len(X)) if classifications[i] == cls and X[i][1] != 0]
        plt.scatter(x_values, y_values, color=class_colors[cls], marker=class_markers[cls], label=cls, alpha=0.7, s=50)

    # Add legend with better formatting
    legend = plt.legend()

    # Add labels and title
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)

    # Add grid lines
    plt.grid(True)

    # Show plot
    plt.show()



