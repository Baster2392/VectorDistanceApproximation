import torch
import torch.nn as nn
import torch.optim as optim

from siamese_model import SiameseNetwork
import vector_generator as vg


def testing():
    # Przygotowanie danych treningowych
    n_samples = 256
    input_dim = 10
    hidden_dim = 10
    max_value = 1000

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Utworzenie instancji modelu
    model = SiameseNetwork(input_dim, hidden_dim)

    # Funkcja straty i optymizator
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.to(device)
    criterion.to(device)

    # Trening modelu
    epochs = 10000
    for epoch in range(epochs):
        x_train, y_train = vg.generate_sample_data(n_samples, 0, max_value, input_dim)

        # Konwersja danych treningowych i testowych na typ Float
        x_train = torch.tensor(x_train, dtype=torch.float)
        y_train = torch.tensor(y_train, dtype=torch.float)

        # przenieś na kartę graficzną
        x_train = x_train.to(device)
        y_train = y_train.to(device)

        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        #print("Calculated by model:", outputs)
        #print("Correct:", y_train)
        #loss.backward()
        #optimizer.step()

        if loss.item() / max_value < 0.005:
            break

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}, Percentage: {(loss.item() / max_value) * 100}%')

    new_input_dim = 10
    model.scale_input_size(new_input_dim)
    x_test, y_test = vg.generate_sample_data(n_samples, 0, max_value, new_input_dim)
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


if __name__ == '__main__':
    testing()

