import numpy as np
from sklearn.decomposition import PCA
import old.data_generators.vector_generator as vg

pca = PCA(n_components=2)

def reduce_pairs(pair):
    pca.fit(pair)
    reduced_pair = pca.transform(pair)
    return reduced_pair


def generate_sample_data(size, min_value, max_value, num_pairs):
    pairs = vg.generate_vector_pairs(min_value, max_value, size, num_pairs)
    reduced_pairs = []
    for pair in pairs:
        reduced_pairs.append(reduce_pairs(pair))
    return pairs, np.asarray(reduced_pairs)


if __name__ == '__main__':
    x, y, = generate_sample_data(10, 0, 1, 5)
    print(x)
    print(y)
    distance_o = np.linalg.norm(x[0, 0, :] - x[0, 1, :])
    distance_r = np.linalg.norm(y[0, 0] - y[0, 1])
    print(distance_o)
    print(distance_r)