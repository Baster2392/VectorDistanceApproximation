import time
import torch
import numpy as np

CSV_FILE_PATH = '../saved_results/res2.csv'


def calculate_accuracy(classifications, y_test):
    correct = 0
    total = len(classifications)

    for predicted_class, true_class in zip(classifications, y_test):
        if predicted_class == true_class:
            correct += 1

    accuracy = (correct / total) * 100
    return accuracy



def classify_samples(model, centroids, X_test):
    classifications = []
    for sample in X_test:
        min_distance = float('inf')
        closest_class = None
        for cls, centroid in centroids.items():
            input_pair = np.array([sample, centroid], dtype=np.float32)
            input_pair = torch.tensor(input_pair).unsqueeze(0)
            output = model(input_pair)
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



def validate(model, criterion, x_validate, y_validate):
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        y_pred = model(x_validate)
        loss = criterion(y_pred, y_validate)
    elapsed_time = time.time() - start_time
    for i in range(len(y_validate)):
        print("Predicted:", y_pred[i], "Actual:", y_validate[i])


    print("Mean loss:", loss.item())
    print("Max loss:", torch.max(abs(y_pred - y_validate)))
    print("Min loss:", torch.min(abs(y_pred - y_validate)))
    print("Min error:", torch.min(abs(y_pred - y_validate) / y_validate))
    print(f"Time Taken: {elapsed_time} seconds")

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
        output = model(x_train)
        loss = custom_loss(output, y_train, similarity_database)  # Use custom loss function
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

