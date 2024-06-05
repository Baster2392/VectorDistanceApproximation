import torch
import torch.nn as nn
import torch.optim as optim
import itertools
from torch.optim.lr_scheduler import ReduceLROnPlateau
import csv

from models.recurrect_model import SimpleRNN
import data_generators.vector_generator as vg

CSV_FILE_PATH = '../results/grid_search.csv'


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
        x_train, y_train = vg.generate_sample_data_for_recurrent(n_samples, 0, 1, model.input_dim)
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
            print(f'Id: {model.input_dim}, Hd: {model.hidden_dim} Lrn: {model.num_layers_recurrent} Lfcn: {model.num_layers_fc} Epoch [{epoch}/{epochs}], Loss: {loss.item()}, Lr: {optimizer.param_groups[0]["lr"]}')

        # Check if function converged
        if loss.item() < loss_tolerance:
            break

    return model, epoch + 1, loss.item(), optimizer.param_groups[0]["lr"]


def grid_search(criterion, optimizer_obj, scheduler_obj, epochs, n_samples, loss_tolerance, device):
    best_epoch = float('inf')
    best_params = None
    scheduler = None

    path = CSV_FILE_PATH  # + "_id_" + str(input_dim) + ".csv"
    with open(path, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)

        for hidden_dim, num_layers_recurrent, num_layers_fc in itertools.product(hidden_dims, num_recurrent_layers_list, num_fc_layers_list):
            model = SimpleRNN(input_dim, hidden_dim, num_layers_recurrent, num_layers_fc)
            optimizer = optimizer_obj(model.parameters(), lr=0.001)
            if scheduler_obj is not None:
                scheduler = scheduler_obj(optimizer, mode="min", patience=300, factor=0.75, verbose=True, min_lr=1e-8)

            model, epoch, loss, out_lr = train(model, criterion, optimizer, scheduler, epochs, n_samples, loss_tolerance, device)
            writer.writerow([input_dim, hidden_dim // input_dim, hidden_dim, out_lr, num_layers_recurrent, epoch, loss])
            print(
                f'Parameters: Hidden Dim={hidden_dim}, Learning Rate={out_lr}, Num Layers={num_layers_recurrent}, Epoch={epoch}, Loss={loss}'
            )

            if epoch < best_epoch:
                best_epoch = epoch
                best_params = {'hidden_dim': hidden_dim, 'learning_rate': out_lr, 'num_layers': num_layers_recurrent,
                               'loss': loss}

    # do not treat the best params as definitive, always consult with csv,
    # sometimes because of early stop mechanisms the best params cause bigger
    # loss then some other parameters
    print(f'Best Parameters: {best_params}, Best Epoch: {best_epoch}')

    return best_params


if __name__ == '__main__':
    CSV_FILE_PATH = '../results/grid_search_recurrent_hidden_dim_32.csv'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.L1Loss()
    optimizer = optim.Adam
    scheduler = None

    input_dims = [20]
    for input_dim in input_dims:
        hidden_dims = [i for i in range(2, 34, 2)]
        learning_rates = [0.001]
        num_recurrent_layers_list = [2]
        num_fc_layers_list = [2]
        for i in range(3):
            print("Loop:", i, " for id=", input_dim)
            best_params = grid_search(criterion, optimizer, scheduler, epochs=20000, n_samples=64, loss_tolerance=0.05, device=device)