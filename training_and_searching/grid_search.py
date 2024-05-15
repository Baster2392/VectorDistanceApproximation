import torch
import torch.nn as nn
import torch.optim as optim
import csv
import itertools
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.siamese_model_no_norm import SiameseNetworkNoNorm
from training import train


def grid_search(criterion, optimizer_obj, scheduler_obj, epochs, n_samples, loss_tolerance, device):
    best_epoch = float('inf')
    best_params = None

    path = CSV_FILE_PATH  # + "_id_" + str(input_dim) + ".csv"
    with open(path, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)

        for hidden_dim, num_layers in itertools.product(hidden_dims, num_layers_list):
            model = SiameseNetworkNoNorm(input_dim, hidden_dim, num_layers)
            optimizer = optimizer_obj(model.parameters(), lr=0.01)
            scheduler = scheduler_obj(optimizer, mode="min", patience=300, factor=0.75, verbose=True, min_lr=1e-8)

            model, epoch, loss, out_lr = train(model, criterion, optimizer, scheduler, epochs, n_samples, loss_tolerance, device)
            writer.writerow([input_dim, hidden_dim // input_dim, hidden_dim, out_lr, num_layers, epoch, loss])
            print(
                f'Parameters: Hidden Dim={hidden_dim}, Learning Rate={out_lr}, Num Layers={num_layers}, Epoch={epoch}, Loss={loss}'
            )

            if epoch < best_epoch:
                best_epoch = epoch
                best_params = {'hidden_dim': hidden_dim, 'learning_rate': out_lr, 'num_layers': num_layers,
                               'loss': loss}

    # do not treat the best params as definitive, always consult with csv,
    # sometimes because of early stop mechanisms the best params cause bigger
    # loss then some other parameters
    print(f'Best Parameters: {best_params}, Best Epoch: {best_epoch}')

    return best_params


if __name__ == '__main__':
    CSV_FILE_PATH = '../results/grid_search.csv'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.L1Loss()
    optimizer = optim.Adam
    scheduler = ReduceLROnPlateau

    input_dims = [10, 15, 20, 25, 35, 50, 60, 70, 80, 90, 100, 125, 150]
    for input_dim in input_dims:
        hidden_dims = [i * input_dim for i in range(2, 16, 2)]
        print(hidden_dims)
        # hidden_dims = [450]
        learning_rates = [0.01]
        num_layers_list = [1]
        for i in range(1):
            print("Loop:", i, " for id=", input_dim)
            best_params = grid_search(criterion, optimizer, scheduler, epochs=25000, n_samples=32, loss_tolerance=0.1, device=device)