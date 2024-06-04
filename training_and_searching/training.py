import glob
import time
import torch
import torch.nn as nn
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import optuna
import matplotlib.colors as mcolors
import pandas as pd
import datetime
from models.siamese_model import SiameseNetwork
from data_generators import vector_generator as vg

CSV_FILE_PATH = '../saved_results/results.csv'
bmodel = None
boptimizer = None
bepoch = 0
bmin_loss = float('inf')


"""
The way this code functions has been altered, but it's still essentially a grid search.
Normally it goes over all the possible combinations of the parameters and trains - each such training is called a study.
Study may consist of many trials, those are the second to last argument of the run_experiments function. Currently it's
made so that only data from the most successful trial is saved for more representative results.

If [checkpoint_dir] is provided each study with the same structure
(ie. same input_dim, hidden_dim, num_siamese_layers, num_shared_layers)
will load the model from previous study and continue training from there. 
That way you can easily and efficiently check for example the relation 
between number of epochs and loss tolerance without going through the 
same number of epochs many times over.


"""










LABEL = "W-1"


def moving_average(data, window_size):
    """Calculate moving average with variable window size at the edges."""
    result = np.convolve(data, np.ones(window_size) / window_size, mode='valid')
    start_avg = [np.mean(data[:i + 1]) for i in range(window_size // 2)]
    end_avg = [np.mean(data[-(i + 1):]) for i in range(window_size // 2, 0, -1)]
    return np.concatenate((start_avg, result, end_avg))
def validate(model, criterion, x_validate, y_validate, datasize, loss_tolerance, string="", metric='euclidean'):
    model.eval()
    start_time = time.time()

    with torch.no_grad():
        y_pred = model(x_validate, metric)
        output = y_pred / y_validate
        ones = y_validate / y_validate
        loss = criterion(output, ones)

    elapsed_time = time.time() - start_time

    y_validate = y_validate.view(-1)
    y_pred = y_pred.view(-1)

    sorted_indices = torch.argsort(y_validate)
    y_validate_sorted = y_validate[sorted_indices]
    y_pred_sorted = y_pred[sorted_indices]

    y_validate_sorted_np = y_validate_sorted.cpu().numpy()
    y_pred_sorted_np = y_pred_sorted.cpu().numpy()

    errors = torch.abs(y_pred - y_validate)
    relative_errors = errors / torch.abs(y_validate)

    mean_loss = loss.item()
    max_loss = torch.max(errors).item()

    #plot_validation_results(y_validate_sorted_np, y_pred_sorted_np, model.input_dim, string, mean_loss, max_loss, datasize, loss_tolerance)

    print_validation_results(mean_loss, max_loss, errors, relative_errors, elapsed_time, x_validate)

    start_time = time.time()
    typical_distances = [vg.calculate_distance(pair[0], pair[1], metric) for pair in x_validate]
    typical_time = time.time() - start_time

    validation_results = {
        'mean_loss': mean_loss,
        'calc_times': (elapsed_time, typical_time),
        'training_size': datasize,
    }

    return validation_results
def plot_validation_results(y_validate_sorted_np, y_pred_sorted_np, input_dim, string, mean_loss, max_loss, datasize, loss_tolerance):
    plt.figure(figsize=(8, 6))
    plt.plot(y_pred_sorted_np, label='Predicted', color='green')

    window_size = 10
    moving_avg = moving_average(y_pred_sorted_np, window_size)
    plt.plot(moving_avg, label=f'Moving Average ({window_size})', color='orange')
    plt.plot(y_validate_sorted_np, label='Actual', color='blue')

    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title(f'Actual vs Predicted {input_dim} : {string}')
    plt.legend()
    plt.grid(True)

    plt.text(0.05, 0.95, f'Mean loss: {mean_loss:.4f}', transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top')
    plt.text(0.05, 0.90, f'Max loss: {max_loss:.4f}', transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top')
    plt.text(0.05, 0.80, f'Datasize: {datasize}', transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top')
    plt.text(0.05, 0.75, f'Loss Tolerance: {loss_tolerance}', transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top')
    plt.legend(loc='lower left')

    plt.show()
def print_validation_results(mean_loss, max_loss, errors, relative_errors, elapsed_time, x_validate, metric='euclidean'):
    print(
        f"\033[33;5;229mMean loss: {mean_loss}, Max loss: {max_loss}, Min loss: {torch.min(errors).item()}, Mean error: {torch.mean(relative_errors).item()}, \nMax error: {torch.max(relative_errors).item()}, Min error: {torch.min(relative_errors).item()}\033[0m")

    start_time = time.time()
    typical_distances = [vg.calculate_distance(pair[0], pair[1], metric) for pair in x_validate]
    typical_time = time.time() - start_time
    print(
        f"\033[33;5;241mTime Taken: {elapsed_time:.4f} seconds || Time Taken Using Traditional Methods: {typical_time:.4f} seconds\033[0m")
def save_model_checkpoint(model, optimizer, epoch, min_loss, path, input_dim, hidden_dim,num_siamese_layers, num_shared_layers):
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'min_loss': min_loss
    }
    path = f"{input_dim}_{hidden_dim}_{num_siamese_layers}_{num_shared_layers}"+path
    torch.save(state, path)
def load_model_checkpoint(path, model, optimizer, input_dim, hidden_dim,num_siamese_layers, num_shared_layers):
    path = f"{input_dim}_{hidden_dim}_{num_siamese_layers}_{num_shared_layers}"+path
    if os.path.isfile(path):
        state = torch.load(path)
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        epoch = state['epoch']
        min_loss = state['min_loss']
        print(f"\033[95mLoaded checkpoint '{path}' (epoch {epoch}, min_loss {min_loss})\033[0m")
        return model, optimizer, epoch, min_loss
    else:
        return model, optimizer, 0, float('inf')
def train(model, criterion, optimizer, scheduler, epochs, n_samples, data_size,
          loss_tolerance=0.5, device=torch.device('cpu'), print_every=10, start_epoch=0, min_loss=float('inf'),
          metric='euclidean', checkpoint_path=None, input_dim=None, hidden_dim=None,num_siamese_layers=None, num_shared_layers=None):
    model.to(device)
    criterion.to(device)
    min_error = float('inf')
    full_x_train, full_y_train = vg.generate_sample_data_with_multithreading(data_size, -1, 1, model.input_dim, metric)
    full_x_train = torch.tensor(full_x_train, dtype=torch.float).to(device)
    full_y_train = torch.tensor(full_y_train, dtype=torch.float).to(device)

    if checkpoint_path:
        model, optimizer, start_epoch, min_loss = load_model_checkpoint(checkpoint_path, model, optimizer, input_dim, hidden_dim,num_siamese_layers, num_shared_layers)

    model.train()

    for epoch in range(start_epoch, epochs):
        try:
            indices = torch.randperm(data_size)[:n_samples]
            x_train = full_x_train[indices]
            y_train = full_y_train[indices]

            optimizer.zero_grad()

            output = model(x_train, metric)
            output = output / y_train
            ones = y_train / y_train
            loss = criterion(output, ones)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            if scheduler is not None:
                scheduler.step(loss.item())

            if epoch % print_every == 0:
                print(
                    f'\rId: {model.input_dim} Epoch [{epoch}/{epochs}], Loss: {loss.item()}, Lr: {optimizer.param_groups[0]["lr"]} Smallest Found Loss: {min_loss}',
                    end="")

            if loss.item() < loss_tolerance:
                break

            if loss.item() < min_loss:
                min_loss = loss.item()
                print(
                    f'\r\033[92mId: {model.input_dim} Epoch [{epoch}/{epochs}], Loss: {loss.item()}, '
                    f'Lr: {optimizer.param_groups[0]["lr"]} Smallest Found Loss: {min_loss} \033[0m:'
                    , end="")
        except Exception as e:
            print(f"Error during training at epoch {epoch}: {e}")
            break
    global bmodel, boptimizer, bepoch, bmin_loss
    if checkpoint_path and min_loss < bmin_loss:
        bmodel= model
        boptimizer = optimizer
        bepoch = epoch
        bmin_loss = min_loss


    print();
    #plot_weights_and_biases(model)

    # Validation
    x_validate = full_x_train
    y_validate = full_y_train
    validation_results = validate(model, criterion, x_validate, y_validate, data_size, loss_tolerance)

    return model, epoch + 1, loss.item(), optimizer.param_groups[0][
        "lr"], min_loss, full_x_train, full_y_train, validation_results
def plot_weights_and_biases(model):
    weights_per_layer = []
    biases_per_layer = []

    # Iterate over the layers in the model
    for name, param in model.named_parameters():
        if 'weight' in name:
            weights_per_layer.append(param.data.cpu().numpy())
        elif 'bias' in name:
            biases_per_layer.append(param.data.cpu().numpy())

    num_layers = len(weights_per_layer)

    # Calculate average output weights for each neuron
    avg_weights = []
    for i in range(num_layers):
        weights = weights_per_layer[i]
        # Calculate mean weight for this layer
        layer_mean_weight = np.mean(weights)
        normalized_weights = weights / layer_mean_weight  # Normalize by layer mean
        avg_weights.append(normalized_weights.mean(axis=1))  # Average over input connections

    plt.figure(figsize=(12, 6))
    for i in range(num_layers):
        avg_weights_layer = avg_weights[i]
        num_neurons = len(avg_weights_layer)
        # Color represents the weight relative to the layer's mean (importance within layer)
        plt.scatter(np.full(num_neurons, i), np.arange(num_neurons), c=avg_weights_layer, cmap='coolwarm', marker='s', s=100, alpha=0.5)

    plt.colorbar(label='Normalized Weight (Importance within Layer)')
    plt.xlabel('Layer Index')
    plt.ylabel('Neuron Index')
    plt.title('Normalized Average Output Weights Distribution (Relative Importance)')
    plt.grid(True)
    plt.show()
def objective(trial, num_siamese_layers, num_shared_layers, hidden_dim, input_dim, epochs, loss_tolerance, data_size, n_samples, metric, mode):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SiameseNetwork(input_dim, hidden_dim, num_siamese_layers, num_shared_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=200, factor=0.75, min_lr=1e-10, verbose=True)
    criterion = nn.MSELoss()

    current_time = datetime.datetime.now()
    model, _, _, _, val_loss, _, _, validation_results = train(model, criterion, optimizer, scheduler, epochs=epochs,
                                                               n_samples=n_samples, loss_tolerance=loss_tolerance,
                                                               device=device, data_size=data_size, print_every=100, metric=metric, checkpoint_path=mode, input_dim=input_dim, hidden_dim=hidden_dim,num_siamese_layers=num_siamese_layers, num_shared_layers=num_shared_layers)
    training_time_delta = datetime.datetime.now() - current_time

    # Access minutes using total_seconds and division
    minutes = int(training_time_delta.total_seconds() / 60)

    # Access remaining seconds (optional)
    seconds = int(training_time_delta.total_seconds() % 60)

    # Access microseconds (optional, similar to seconds)
    microseconds = training_time_delta.microseconds

    # Formatting the time string
    training_time_str = f"{minutes}:{seconds}.{microseconds}"
    mean_loss = validation_results['mean_loss']
    calc_times = validation_results['calc_times']
    training_size = validation_results['training_size']
    label = LABEL

    trial.set_user_attr('mean_loss', mean_loss)
    trial.set_user_attr('calc_times', calc_times)
    trial.set_user_attr('training_size', training_size)
    trial.set_user_attr('label', label)
    trial.set_user_attr('device', str(device))
    trial.set_user_attr('time', current_time.strftime("%Y-%m-%d %H:%M"))
    trial.set_user_attr('training time', training_time_str)
    trial.set_user_attr('metric', metric)


    return val_loss
def run_experiments(hidden_dims, num_siamese_layers_values, num_shared_layers_values, input_dims, epochs_values, loss_tolerance_values, data_size_values, n_samples_values, metrics,trials=1, mode=None):
    results = []
    for metric in metrics:
        for num_siamese_layers in num_siamese_layers_values:
            for num_shared_layers in num_shared_layers_values:
                for input_dim in input_dims:
                    for epochs in epochs_values:
                        for loss_tolerance in loss_tolerance_values:
                            for data_size in data_size_values:
                                for n_samples in n_samples_values:
                                    trial_results = []
                                    for hidden_dim in hidden_dims:
                                        if mode:
                                            global bmodel, boptimizer, bepoch, bmin_loss
                                            bmodel = None
                                            boptimizer = None
                                            bepoch = 0
                                            bmin_loss = float('inf')
                                        study = optuna.create_study(direction='minimize')
                                        study.optimize(lambda trial: objective(trial, num_siamese_layers, num_shared_layers, hidden_dim, input_dim, epochs, loss_tolerance, data_size, n_samples, metric, mode), n_trials=trials)
                                        trial = study.best_trial
                                        min_loss = trial.value
                                        user_attrs = trial.user_attrs
                                        if mode:
                                            save_model_checkpoint(bmodel, boptimizer, bepoch, bmin_loss, mode, input_dim, hidden_dim,num_siamese_layers, num_shared_layers)
                                        trial_results.append((min_loss, user_attrs))
                                        print(
                                            f"num_siamese_layers: {num_siamese_layers}, num_shared_layers: {num_shared_layers}, input_dim: {input_dim}, hidden_dim: {hidden_dim}, epochs: {epochs}, loss_tolerance: {loss_tolerance}, data_size: {data_size}, n_samples: {n_samples}, min_loss: {min_loss}")
                                        print(f'\033[38;5;208m====================================\033[0m:')
                                    results.append((num_siamese_layers, num_shared_layers, input_dim, epochs, loss_tolerance, data_size, n_samples, trial_results))

    return results
def create_custom_colormap(min_value, max_value):
    # Define colors for the colormap
    colors = [(0, 0.5, 0), (0.6, 0.8, 0.2)]  # Dark green to light green
    # Define the levels for the colormap based on the min and max values
    levels = np.linspace(min_value, max_value, 256)
    # Create the colormap
    cmap = mcolors.LinearSegmentedColormap.from_list('custom_colormap', colors, N=len(levels))
    return cmap
def save_experiment_results(results, hidden_dims):
    # Create a list to store the results
    data = []

    for (num_siamese_layers, num_shared_layers, input_dim, epochs, loss_tolerance, data_size, n_samples, trials) in results:
        for hidden_dim, (loss, user_attrs) in zip(hidden_dims, trials):
            row = [
                num_siamese_layers,
                num_shared_layers,
                input_dim,
                hidden_dim,
                loss,
                epochs,
                loss_tolerance,
                data_size,
                n_samples,
                user_attrs.get('mean_loss', np.nan),
                user_attrs.get('calc_times', (np.nan, np.nan))[0],
                user_attrs.get('calc_times', (np.nan, np.nan))[1],
                user_attrs.get('training_size', np.nan),
                user_attrs.get('label', ""),
                user_attrs.get('device', ""),
                user_attrs.get('time', ""),
                user_attrs.get('training time', ""),
                user_attrs.get('metric', "")
            ]
            data.append(row)

    # Create a DataFrame from the data
    df1 = pd.DataFrame(data, columns=[
        'Siamese Layers', 'Shared Layers', 'Input Dim', 'Hidden Dimension', 'Train Loss',
        'Epochs', 'Loss Tolerance', 'Datasize', 'n_samples', 'Test Loss',
        'Calc Time (Model)', 'Calc Time (Traditional)', 'Training Size', 'Label', 'Device', 'Time', 'Training Time', 'metric'
    ])

    # Check if results.csv exists
    if os.path.isfile(CSV_FILE_PATH):
        # If it exists, read the existing data
        df = pd.read_csv(CSV_FILE_PATH, index_col=None)
    else:
        # If it doesn't exist, create an empty DataFrame
        df = pd.DataFrame(columns=[
            'Siamese Layers', 'Shared Layers', 'Input Dim', 'Hidden Dimension', 'Train Loss',
            'Epochs', 'Loss Tolerance', 'Datasize', 'n_samples', 'Test Loss',
            'Calc Time (Model)', 'Calc Time (Traditional)', 'Training Size', 'Label', 'Device', 'Time', 'Training Time',
            'metric'
        ])

    # Append the new data to the DataFrame
    df = pd.concat([df, df1], ignore_index=True)

    # Save the DataFrame to the CSV file
    df.to_csv(CSV_FILE_PATH, index=False)


def delete_checkpoint_files(checkpoint_dir):
  """
  This function deletes files containing 'checkpoint_dir' in their path.

  Args:
    checkpoint_dir: The string to search for in file paths.
  """
  files_to_delete = glob.glob(f"**/*{checkpoint_dir}*", recursive=True)
  if files_to_delete:
    print(f"Deleting files containing '{checkpoint_dir}':")
    for file in files_to_delete:
      print(file)
      os.remove(file)
    print("Files deleted successfully!")
  else:
    print(f"No files found containing '{checkpoint_dir}'.")

def get_device_and_print_info():
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  color_code = {
      'cuda': "\033[92m",  # Green for CUDA
      'cpu': "\033[91m"    # Red for CPU
  }

  print(f"{color_code[device.type]}==================\nUsing device:{device}\n==================\033[0m")

  return device

if __name__ == '__main__':
    device = get_device_and_print_info()

    # crucial values - these values determine the topology of the network
    hidden_dims = [500]
    num_siamese_layers_values = [2]
    num_shared_layers_values = [5]
    input_dims = [100]


    epochs_values = [2000]
    loss_tolerance_values = [0.001]
    data_size_values = [20000]
    n_samples_values = [64]

    metrics = ['euclidean']
    checkpoint_dir = "checkpoints"



    results = run_experiments(hidden_dims, num_siamese_layers_values, num_shared_layers_values, input_dims,
                              epochs_values, loss_tolerance_values, data_size_values, n_samples_values, metrics, 1, checkpoint_dir)
    save_experiment_results(results, hidden_dims)

    delete_checkpoint_files(checkpoint_dir)


