import random

import numpy
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


def generate_vector(min_value, max_value, size, for_recurrent=False):
    if not for_recurrent:
        vector = numpy.array([abs(np.random.randn()) * (max_value - min_value) + min_value for _ in range(size)])
    else:
        ran = np.random.randint(2, size + 1)
        vector = numpy.array(
            [[abs(np.random.randn()) * (max_value - min_value) + min_value] for _ in range(size - ran)] +
            [[0] for _ in range(ran)]
        )
    return vector


def generate_random_size_vector(min_value, max_value, smallest_dim_size, highest_dim_size):
    vector = numpy.zeros((highest_dim_size, 1))
    for i in range(numpy.random.randint(smallest_dim_size, highest_dim_size)):
        vector[i, 0] = abs(numpy.random.randn()) * (max_value - min_value) + min_value
    return vector


def generate_vector_pair(min_value, max_value, vector_size, for_recurrent=False):
    if not for_recurrent:
        vector_pair = np.zeros((2, vector_size))
        vector_pair[0] = generate_vector(min_value, max_value, vector_size)
        vector_pair[1] = generate_vector(min_value, max_value, vector_size)
    else:
        vector_pair = np.zeros((2, vector_size, 1))
        vector_pair[0] = generate_vector(min_value, max_value, vector_size, for_recurrent=True)
        vector_pair[1] = generate_vector(min_value, max_value, vector_size, for_recurrent=True)
    return vector_pair


def generate_random_size_vector_pair(min_value, max_value, smallest_dim_size, highest_dim_size):
    vector_pair = np.zeros((2, highest_dim_size, 1))
    vector_pair[0, :] = generate_random_size_vector(min_value, max_value, smallest_dim_size, highest_dim_size)
    vector_pair[1, :] = generate_random_size_vector(min_value, max_value, smallest_dim_size, highest_dim_size)
    return vector_pair


def generate_vector_pairs(min_value, max_value, vector_size, pairs_number, for_recurrent=False):
    if not for_recurrent:
        vector_pairs = np.zeros((pairs_number, 2, vector_size))
        for i in range(pairs_number):
            vector_pairs[i] = generate_vector_pair(min_value, max_value, vector_size)
    else:
        vector_pairs = np.zeros((pairs_number, 2, vector_size, 1))
        for i in range(pairs_number):
            vector_pairs[i] = generate_vector_pair(min_value, max_value, vector_size, for_recurrent=True)
    return vector_pairs


def generate_vector_pairs_recurrent_siamese(min_value, max_value, pairs_number, smallest_dim_size, highest_dim_size):
    pairs = numpy.zeros((pairs_number, 2, highest_dim_size, 1))
    for i in range(pairs_number):
        pairs[i, :, :] = generate_random_size_vector_pair(min_value, max_value, smallest_dim_size, highest_dim_size)
    return pairs


def calculate_distance(a, b):
    return np.linalg.norm(a - b)


def generate_sample_data(number_of_samples, min_value, max_value, vector_size, split_pairs=False):
    if not split_pairs:
        sample_pairs = generate_vector_pairs(min_value, max_value, vector_size, number_of_samples)
        sample_distances = np.zeros((number_of_samples, 1))
        for i in range(number_of_samples):
            sample_distances[i] = calculate_distance(sample_pairs[i][0], sample_pairs[i][1])
        return sample_pairs, sample_distances
    else:
        lefts, rights, distances = [], [], []
        for i in range(number_of_samples):
            lefts.append(generate_vector_pair(min_value, max_value, vector_size))
            rights.append(generate_vector_pair(min_value, max_value, vector_size))
            distances.append(calculate_distance(lefts[i], rights[i]))
        return lefts, rights, distances


def generate_sample_data_for_recurrent_siamese(number_of_samples, min_value, max_value, smallest_dim_size, highest_dim_size):
    sample_pairs = generate_vector_pairs_recurrent_siamese(min_value, max_value, number_of_samples, smallest_dim_size, highest_dim_size)
    sample_distances = np.zeros((number_of_samples, 1))
    for i in range(number_of_samples):
        sample_distances[i] = calculate_distance(sample_pairs[i][0], sample_pairs[i][1])
    return sample_pairs, sample_distances


def generate_sample_data_for_recurrent(number_of_samples, min_value, max_value, vector_size):
    sample_pairs = generate_vector_pairs(min_value, max_value, vector_size, number_of_samples, for_recurrent=True)
    x1 = torch.tensor(sample_pairs[:, 0, :, :])
    x2 = torch.tensor(sample_pairs[:, 1, :, :])
    sample_distances = torch.zeros((number_of_samples, 1))
    for i in range(number_of_samples):
        sample_distances[i] = calculate_distance(sample_pairs[i][0], sample_pairs[i][1])
    return torch.cat((x1, x2), dim=2), sample_distances


# tests
if __name__ == '__main__':
    x_data, y_data = generate_sample_data_for_recurrent(10, 0, 1, 2)
    for line1, line2 in zip(x_data, y_data):
        print(line1, line2)

