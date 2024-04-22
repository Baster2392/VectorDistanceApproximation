import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import vector_generator as vg


class SiameseNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SiameseNetwork, self).__init__()

        # Definicja identycznych gałęzi sieci
        self.branch1 = self.create_branch(input_dim, hidden_dim)
        self.branch2 = self.create_branch(input_dim, hidden_dim)
        self.shared_branch1 = self.create_branch(2 * hidden_dim, hidden_dim)
        self.shared_branch2 = self.create_branch(hidden_dim, hidden_dim)

        # Warstwa przekształcająca wyniki z obu gałęzi do jednego wymiaru
        self.fc = nn.Linear(hidden_dim, 1)

    def create_branch(self, input_dime, hidden_dime):
        return nn.Sequential(
            nn.Linear(input_dime, hidden_dime),
            nn.ReLU(),
            nn.Linear(hidden_dime, hidden_dime),
            nn.ReLU()
        )

    def forward(self, x):
        # Przekształcenie obu wejściowych wektorów przez gałęzie sieci
        x1 = self.branch1(x[:, 0, :])
        x2 = self.branch2(x[:, 1, :])

        # Wejście do warstwy współdzielonej
        combined_x = torch.cat((x1, x2), dim=1)
        shared_out = self.shared_branch1(combined_x)

        # Propagacja przez warstwę przekształcającą
        out = self.fc(shared_out)
        return out


# Przygotowanie danych treningowych
n_samples = 1000
input_dim = 20000
hidden_dim = 128
max_value = 1000

# generowanie danych
x_train, y_train = vg.generate_sample_data(n_samples, 0, max_value, input_dim)
x_test, y_test = vg.generate_sample_data(n_samples, 0, max_value, input_dim)

# Konwersja danych treningowych i testowych na typ Float
x_train = torch.tensor(x_train, dtype=torch.float)
y_train = torch.tensor(y_train, dtype=torch.float)
x_test = torch.tensor(x_test, dtype=torch.float)
y_test = torch.tensor(y_test, dtype=torch.float)

# Utworzenie instancji modelu
model = SiameseNetwork(input_dim, hidden_dim)

# Funkcja straty i optymizator
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Trening modelu
epochs = 250
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(x_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}, Percentage: {(loss.item() / max_value) * 100}%')

exit(0)
# Testowanie modelu
with torch.no_grad():
    test_outputs = model(x_test)
    print("Test outputs:")
    for i in range(test_outputs.size()[0]):
        print(test_outputs[i], y_test[i])
