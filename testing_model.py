import torch
from main import SiameseNetwork
import vector_generator as vg
import numpy as np

input_dim = 1000
hidden_dim = 128

# Tworzenie nowej instancji modelu
model = SiameseNetwork(input_dim, hidden_dim)

# Ładowanie wytrenowanych wag z pliku
model.load_state_dict(torch.load('siamese_model.pth'))

pair, distance = vg.generate_sample_data(1, 0, 10000, input_dim)
pair = torch.tensor(pair, dtype=torch.float)

if __name__ == '__main__':
    print(pair)
    calc_dist = model.forward(pair)
    print("Calculated distance is: ", calc_dist)
    print("True distance is: ", distance)
    print("Difference is: ", abs(calc_dist - distance[0]))

