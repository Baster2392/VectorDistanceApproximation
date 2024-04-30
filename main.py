import torch
import torch.nn as nn
import torch.optim as optim

from siamese_model import SiameseNetwork
import vector_generator as vg


def trening(input_dim, hidden_dim):
    # Przygotowanie danych treningowych
    n_samples = 32
    max_value = 1

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Utworzenie instancji modelu
    model = SiameseNetwork(input_dim, hidden_dim).to(device)

    # Funkcja straty i optymizator
    criterion = nn.L1Loss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    # Trening modelu
    epochs = 100000
    for epoch in range(epochs):
        x_train, y_train = vg.generate_sample_data(n_samples, 0, max_value, input_dim)

        # Konwersja danych treningowych i testowych na typ Float
        x_train = torch.tensor(x_train, dtype=torch.float).to(device)
        y_train = torch.tensor(y_train, dtype=torch.float).to(device)

        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        # print("Calculated by model:", outputs)
        # print("Correct:", y_train)
        loss.backward()
        optimizer.step()

        if loss.item() < 0.01:
            break

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}')
        # print(f'Epoch [{epoch + 1}/{epochs}], {torch.norm(y_train - outputs).item()}')

    return model


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


if __name__ == '__main__':
    input_dim = 10
    hidden_dim = 10
    model = trening(input_dim, hidden_dim)
    print("Testing model...")
    testing(model, n_samples=20, input_dim=input_dim)

