import torch
from main import SiameseNetwork
import vector_generator as vg
import numpy as np

input_dim = 1000
hidden_dim = 128


pair, distance = vg.generate_sample_data(1, 0, 10000, input_dim)
pair = torch.tensor(pair, dtype=torch.float)

if __name__ == '__main__':
    x_train, y_train = vg.generate_sample_data(3, 0, 1, 5)
    x_train = torch.tensor(x_train, dtype=torch.float)
    y_train = torch.tensor(y_train, dtype=torch.float)
    print(y_train)
    print("")
    print(SiameseNetwork.loss_function(y_train, y_train))

