import torch
import torch.nn as nn
import torch.optim as optim
import csv
import itertools

from siamese_model import SiameseNetwork
import vector_generator as vg

CSV_FILE_PATH = 'training_results.csv'


def trening(input_dim, hidden_dim, learning_rate, num_layers, patience=1000):
    #patience is the number of epochs we wait before checking for early stop
    loss_threshold = learning_rate/10
    n_samples = 32
    max_value = 1

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = SiameseNetwork(input_dim, hidden_dim, num_layers).to(device)
    criterion = nn.L1Loss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    prev_losses = []
    epochs = 100000
    for epoch in range(epochs):
        x_train, y_train = vg.generate_sample_data(n_samples, 0, max_value, input_dim)
        x_train = torch.tensor(x_train, dtype=torch.float).to(device)
        y_train = torch.tensor(y_train, dtype=torch.float).to(device)

        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        # best to keep commented when using grid search
        # Print loss for each epoch
        #print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}')

        if loss.item() < 0.01:
            break




        prev_loss = loss.item() if epoch > 0 else float('inf')
        prev_losses.append(loss.item())
        if epoch >= patience:
            avg_loss_change = sum((prev_losses[i] - prev_losses[i - 1]) for i in range(1, patience)) / patience
            if abs(avg_loss_change) < loss_threshold:
                print(f'Converged Badly! Average loss change: {avg_loss_change}')
                break
            prev_losses.pop(0)

       

    return model, epoch+1, loss.item()



def testing(model, n_samples, input_dim):
    max_value = 1

    x_test, y_test = vg.generate_sample_data(n_samples, 0, max_value, input_dim)
    x_test = torch.tensor(x_test, dtype=torch.float)
    y_test = torch.tensor(y_test, dtype=torch.float)
    torch.save(model.state_dict(), 'siamese_model.pth')
    # Testowanie modelu
    model.to('cpu')
    x_test.to('cpu')
    y_test.to('cpu')
    with torch.no_grad():
        test_outputs = model(x_test)
        print("Test outputs:")
        print(test_outputs.shape)
        for i in range(test_outputs.shape[0]):
            print(test_outputs[i], y_test[i])


def grid_search(input_dim, hidden_dims, learning_rates, num_layers_list):
    best_epoch = float('inf')
    best_params = None

    with open(CSV_FILE_PATH, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Hidden Dim', 'Learning Rate', 'Num Layers', 'Epoch', 'Loss'])

        for hidden_dim, learning_rate, num_layers in itertools.product(hidden_dims, learning_rates, num_layers_list):
            model, epoch, loss = trening(input_dim, hidden_dim, learning_rate, num_layers)
            writer.writerow([hidden_dim, learning_rate, num_layers, epoch, loss])
            print(
                f'Parameters: Hidden Dim={hidden_dim}, Learning Rate={learning_rate}, Num Layers={num_layers}, Epoch={epoch}, Loss={loss}')

            if epoch < best_epoch:
                best_epoch = epoch
                best_params = {'hidden_dim': hidden_dim, 'learning_rate': learning_rate, 'num_layers': num_layers, 'loss':loss}
# do not treat the best params as definitive, always consult with csv,
# sometimes because of early stop mechanisms the best params cause bigger
# loss then some other parameters
    print(f'Best Parameters: {best_params}, Best Epoch: {best_epoch}')
    return best_params


if __name__ == '__main__':
    input_dim = 10
    hidden_dims = [160, 170, 180, 200]
    learning_rates = [0.0001, 0.001]
    num_layers_list = [1, 2, 3]

    best_params = grid_search(input_dim, hidden_dims, learning_rates, num_layers_list)


    #hidden_dim = 1400
    #model, epoch, loss = trening(input_dim, hidden_dim, 0.00001,4)
