import csv
import itertools
import time

import torch
import torch.nn as nn
import torch.optim as optim

import data_generators.vector_generator as vg
from models.recurrect_model import LSTMModel

CSV_FILE_PATH = '../results/grid_search.csv'

def validate(model, criterion, vector_size):
    model.eval()
    x_validate, y_validate = vg.generate_sample_data_for_recurrent(100, 0, 1, vector_size, True)
    x_validate = torch.tensor(x_validate, dtype=torch.float)
    y_validate = torch.tensor(y_validate, dtype=torch.float)

    with torch.no_grad():
        y_pred = model(x_validate)
        loss = criterion(y_pred, y_validate)
    for i in range(len(y_validate)):
        print("Predicted:", y_pred[i], "Actual:", y_validate[i])
    print("Mean loss:", loss.item())
    print("Max loss:", torch.max(abs(y_pred - y_validate)))
    print("Min loss:", torch.min(abs(y_pred - y_validate)))
    return loss.item()


def train(model, criterion, optimizer, scheduler, epochs, n_samples,
          loss_tolerance=0.5, device=torch.device('cpu')):
    # Transfer components to device
    model.to(device)
    criterion.to(device)

    # Generate data
    train_set_size = 10000
    print("Generating data...")
    vector_train_set = vg.generate_vector_set(train_set_size, 0, 1, model.input_dim, True, False)
    vector_train_set = torch.tensor(vector_train_set, dtype=torch.float)

    # Training loop
    model.train()
    epoch = 0
    loss = 0
    min_loss = float('inf')
    for epoch in range(epochs):
        # Generate training data
        x_train, y_train = vg.load_random_batch(vector_train_set, n_samples)

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
            print(f'Id: {model.input_dim}, Hdr: {model.hidden_dim_r}, Hdf: {model.hidden_dim_fc}, Lrn: {model.num_layers_recurrent}, Lfcn: {model.num_layers_fc} Epoch [{epoch}/{epochs}], Loss: {loss.item()}, Lr: {optimizer.param_groups[0]["lr"]} Best loss: {min_loss}')

        # Check if function converged
        if loss.item() < loss_tolerance:
            break

        if loss.item() < min_loss:
            min_loss = loss.item()
            print(f"!!! Found new min_loss {min_loss} in epoch {epoch + 1} !!!")

    return model, epoch + 1, loss.item(), optimizer.param_groups[0]["lr"]


def grid_search(criterion, optimizer_obj, scheduler_obj, hidden_dims_r, hidden_dims_fc, num_recurrent_layers_list, num_fc_layers_list, epochs, n_samples, lr, loss_tolerance, device):
    scheduler, out_lr = None, None
    path = CSV_FILE_PATH
    with open(path, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)

        for hidden_dim, hidden_dim_fc, num_layers_recurrent, num_layers_fc in itertools.product(hidden_dims_r, hidden_dims_fc, num_recurrent_layers_list, num_fc_layers_list):
            model = LSTMModel(input_dim, hidden_dim, hidden_dim_fc=hidden_dim_fc, num_layers_recurrent=num_layers_recurrent, num_layers_fc=num_layers_fc)
            optimizer = optimizer_obj(model.parameters(), lr=lr)
            if scheduler_obj is not None:
                scheduler = scheduler_obj(optimizer, mode="min", patience=300, factor=0.75, verbose=True, min_lr=1e-8)

            # train model
            try:
                model, epoch, loss, out_lr = train(model, criterion, optimizer, scheduler, epochs, n_samples, loss_tolerance, device)
            except KeyboardInterrupt:
                print('Closing training early...')
            finally:
                torch.save(model.state_dict(), f"../saved_models/{model.input_dim}_recurrent_{time.time()}.pth")

            validate_value = validate(model, criterion, input_dim)
            writer.writerow([input_dim, hidden_dim // input_dim, hidden_dim, hidden_dim_fc, out_lr, num_layers_recurrent, num_layers_fc, epoch, loss, validate_value])


if __name__ == '__main__':
    CSV_FILE_PATH = '../results/12loss_tolerance.csv'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.L1Loss()
    optimizer = optim.Adam
    scheduler = None

    input_dims = [100]
    for input_dim in input_dims:
        hidden_dims_r = [64]
        hidden_dims_fc = [300]
        learning_rate = 0.0005
        num_recurrent_layers_list = [2]
        num_fc_layers_list = [3]
        for i in range(1):
            print("Loop:", i, " for id=", input_dim)
            grid_search(criterion, optimizer, scheduler, hidden_dims_r, hidden_dims_fc,
                                      num_recurrent_layers_list, num_fc_layers_list,
                                      epochs=5000000, n_samples=64, lr=learning_rate, loss_tolerance=0.0001, device=device)
