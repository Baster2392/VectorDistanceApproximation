import torch
from old.data_generators import vector_generator as vg
from timeit import default_timer as time
from old.models.recurrect_model import LSTMModel
from old.models.siamese_model import SiameseNetwork
import matplotlib.pyplot as plt

if __name__ == '__main__':
    input_sizes = [100, 500, 1000, 5000, 10000, 50000]
    hidden_size = 64
    batch_size = 32
    recurrent_times = []
    siamese_times = []

    for input_size in input_sizes:
        rnn_model = LSTMModel(input_size, hidden_size)
        siamese_model = SiameseNetwork(input_size, hidden_size, 1, 1)

        data_recurrent_x, data_recurrent_y = vg.generate_sample_data_for_recurrent(batch_size, 0, 1, input_size, True)
        data_siamese_x, data_siamese_y = vg.generate_sample_data(batch_size, 0, 1, input_size)
        data_recurrent_x = torch.tensor(data_recurrent_x, dtype=torch.float)
        data_siamese_x = torch.tensor(data_siamese_x, dtype=torch.float)

        rnn_model.eval()
        siamese_model.eval()
        start_time = time()
        output = rnn_model(data_recurrent_x)
        recurrent_time = time() - start_time
        start_time = time()
        output = siamese_model(data_siamese_x)
        siamese_time = time() - start_time

        recurrent_times.append(recurrent_time)
        siamese_times.append(siamese_time)

    print(recurrent_times)
    print(siamese_times)

    input_sizes_str = []
    for input_size in input_sizes:
        input_sizes_str.append(str(input_size))

    plt.bar(input_sizes_str, recurrent_times, width=1, edgecolor="black", linewidth=0.7, label="Recurrent")
    plt.bar(input_sizes_str, siamese_times, width=1, edgecolor="black", linewidth=0.7, label="Siamese")
    plt.yscale("log")
    plt.xlabel("Input size")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.title("Time Comparison")
    plt.show()
