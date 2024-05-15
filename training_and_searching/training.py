import torch
import torch.nn as nn
import torch.optim as optim
import csv
import itertools
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.siamese_model_no_norm import SiameseNetworkNoNorm
from data_generators import vector_generator as vg

CSV_FILE_PATH = '../saved_results/res2.csv'


def validate(model, criterion, x_validate, y_validate):
    model.eval()
    with torch.no_grad():
        y_pred = model(x_validate)
        loss = criterion(y_pred, y_validate)
    for i in range(len(y_validate)):
        print("Predicted:", y_pred[i], "Actual:", y_validate[i])
    print("Mean loss:", loss.item())
    print("Max loss:", torch.max(abs(y_pred - y_validate)))
    print("Min loss:", torch.min(abs(y_pred - y_validate)))


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

    model = SiameseNetworkNoNorm(3, 20, 1)
    criterion = nn.L1Loss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=600, factor=0.75, min_lr=1e-8, verbose=True)

    model, _, _, _ = train(model, criterion, optimizer, scheduler, epochs=25000, n_samples=32, loss_tolerance=0.05, device=device)

    x_validate, y_validate = vg.generate_sample_data(32, 0, 100, model.input_dim, False)
    x_validate = torch.tensor(x_validate, dtype=torch.float).to(device)
    y_validate = torch.tensor(y_validate, dtype=torch.float).to(device)

    validate(model, criterion, x_validate, y_validate)