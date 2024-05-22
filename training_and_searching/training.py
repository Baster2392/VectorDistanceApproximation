import time
import torch
import torch.nn as nn
import torch.optim as optim
import csv
import itertools
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt

from models.siamese_model import SiameseNetwork
from data_generators import vector_generator as vg

CSV_FILE_PATH = '../saved_results/res2.csv'

class CosineDistance(nn.Module):
    def __init__(self):
        super(CosineDistance, self).__init__()

    def forward(self, x1, x2):
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        cosine_similarities = cos(x1, x2)
        return 1 - torch.mean(cosine_similarities)  # Ensure scalar output

def moving_average(data, window_size):
    """Calculate moving average with variable window size at the edges."""
    result = np.convolve(data, np.ones(window_size) / window_size, mode='valid')
    start_avg = [np.mean(data[:i + 1]) for i in range(window_size // 2)]
    end_avg = [np.mean(data[-(i + 1):]) for i in range(window_size // 2, 0, -1)]
    return np.concatenate((start_avg, result, end_avg))


def validate(model, criterion, x_validate, y_validate):
    model.eval()
    start_time = time.time()

    # Perform prediction and loss calculation without gradient tracking
    with torch.no_grad():
        y_pred = model(x_validate)
        loss = criterion(y_pred, y_validate)

    elapsed_time = time.time() - start_time

    # Print predicted and actual values
    for i in range(len(y_validate)):
        print("Predicted:", y_pred[i].item(), "Actual:", y_validate[i].item())

    # Ensure y_validate and y_pred are 1D
    y_validate = y_validate.view(-1)
    y_pred = y_pred.view(-1)

    # Sorting by actual values
    sorted_indices = torch.argsort(y_validate)
    y_validate_sorted = y_validate[sorted_indices]
    y_pred_sorted = y_pred[sorted_indices]

    # Convert tensors to numpy arrays for plotting
    y_validate_sorted_np = y_validate_sorted.cpu().numpy()
    y_pred_sorted_np = y_pred_sorted.cpu().numpy()

    # Plotting predicted vs actual values
    plt.figure(figsize=(8, 6))
    plt.plot(y_pred_sorted_np, label='Predicted', color='green')

    # Calculate and plot moving average
    window_size = 25
    moving_avg = moving_average(y_pred_sorted_np, window_size)
    plt.plot(moving_avg, label=f'Moving Average ({window_size})', color='orange')
    plt.plot(y_validate_sorted_np, label='Actual', color='blue')


    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Actual vs Predicted')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Calculate errors
    errors = torch.abs(y_pred - y_validate)
    relative_errors = errors / torch.abs(y_validate)

    # Print error metrics
    print("Mean loss:", loss.item())
    print("Max loss:", torch.max(errors).item())
    print("Min loss:", torch.min(errors).item())
    print("Mean error:", torch.mean(relative_errors).item())
    print("Max error:", torch.max(relative_errors).item())
    print("Min error:", torch.min(relative_errors).item())
    print(f"Time Taken: {elapsed_time:.4f} seconds")

    # Assuming vg.calculate_distance is a function that calculates some distance between pairs in x_validate
    start_time = time.time()
    typical_distances = [vg.calculate_distance(pair[0], pair[1]) for pair in x_validate]
    typical_time = time.time() - start_time
    print(f"Time Taken Using Traditional Methods: {typical_time:.4f} seconds")




def train(model, criterion, optimizer, scheduler, epochs, n_samples,
          loss_tolerance=0.5, device=torch.device('cpu')):
    # Transfer components to device
    model.to(device)
    criterion.to(device)

    # Training loop
    model.train()
    epoch = 0
    loss = 0
    for epoch in range(epochs):
        # Generate training data
        x_train, y_train = vg.generate_sample_data(n_samples, 0, 1, model.input_dim, False)
        x_train = torch.tensor(x_train, dtype=torch.float).to(device)
        y_train = torch.tensor(y_train, dtype=torch.float).to(device)

        # Calculate loss
        optimizer.zero_grad()
        output = model(x_train)
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


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SiameseNetwork(10, 100, 1)
    #criterion = CosineDistance()
    criterion = nn.L1Loss(reduction='mean')

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=600, factor=0.75, min_lr=1e-8, verbose=True)

    model, _, _, _ = train(model, criterion, optimizer, scheduler, epochs=1000, n_samples=32, loss_tolerance=0.001 , device=device)

    x_validate, y_validate = vg.generate_sample_data(1000, 0, 100000, model.input_dim, False)
    x_validate = torch.tensor(x_validate, dtype=torch.float).to(device)
    y_validate = torch.tensor(y_validate, dtype=torch.float).to(device)

    validate(model, criterion, x_validate, y_validate)