import numpy
import numpy as np
import torch


def generate_vector(min_value, max_value, size):
    vector = numpy.array([np.random.randn() * (max_value - min_value) + min_value for _ in range(size)])
    return vector


def generate_vector_pair(min_value, max_value, vector_size):
    vector_pair = np.zeros((2, vector_size))
    vector_pair[0] = generate_vector(min_value, max_value, vector_size)
    vector_pair[1] = generate_vector(min_value, max_value, vector_size)
    return vector_pair


def generate_vector_pairs(min_value, max_value, vector_size, pairs_number):
    vector_pairs = np.zeros((pairs_number, 2, vector_size))
    for i in range(pairs_number):
        vector_pairs[i] = generate_vector_pair(min_value, max_value, vector_size)
    return vector_pairs


def calculate_distance(vector1, vector2):
    return np.linalg.norm(vector1 - vector2)


def generate_sample_data(number_of_samples, min_value, max_value, vector_size):
    sample_pairs = generate_vector_pairs(min_value, max_value, vector_size, number_of_samples)
    sample_distances = np.zeros((number_of_samples, 1))
    for i in range(number_of_samples):
        sample_distances[i] = calculate_distance(sample_pairs[i][0], sample_pairs[i][1])
    return sample_pairs, sample_distances


# tests
if __name__ == '__main__':
    data_train, dist_train = generate_sample_data(2, 1, 10, 2)
    data_train = torch.tensor(data_train)
    dist_train = torch.tensor(dist_train)
    print(data_train, data_train.shape)
    print(torch.linalg.norm(data_train[:, 0, :] - data_train[:, 1, :], dim=1, keepdim=True))
    print(dist_train)


