import numpy
import numpy as np
import torch
import concurrent.futures

"""
The distance calculation functionality has been improved to offer greater flexibility. 
It now accepts a metric argument, allowing you to choose the appropriate distance measure:

Euclidean distance ('euclidean') - default
Cosine similarity ('cosine')
Manhattan distance ('manhattan')
Chebyshev distance ('chebyshev')

Also new function, generate_sample_data_with_multithreading, has been added 
to address the challenge of generating large datasets efficiently.

From initial testing I conduct that the backwards compatibility has been maintained,
particularly with the SimpleRNN class.

"""




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


def calculate_distance(a, b, metric='euclidean'):
  """
  Calculates the distance between two vectors.

  Args:
      a: First vector. Can be a NumPy array or PyTorch tensor.
      b: Second vector. Can be a NumPy array or PyTorch tensor.
      metric (str, optional): The distance metric to use.
          Options are 'euclidean' (default), 'cosine', 'manhattan', 'chebyshev'.

  Returns:
      float: The distance between the two vectors.
  """
  # Move tensors to CPU if necessary
  if isinstance(a, torch.Tensor) and a.device.type == 'cuda':
      a = a.cpu()
  if isinstance(b, torch.Tensor) and b.device.type == 'cuda':
      b = b.cpu()



  # Flatten the arrays (optional, might be unnecessary depending on usage)
  # a_flat = np.ravel(a)
  # b_flat = np.ravel(b)

  if metric == 'euclidean':
      # Euclidean distance
      return np.linalg.norm(a - b)
  elif metric == 'cosine':
      # Cosine similarity (distance between 0 and 2, lower means more similar)
      # Handle cases where either vector is all zeros (undefined cosine similarity)
      if np.all(a == 0) or np.all(b == 0):
          return 1.0  # Consider them very dissimilar
      else:
          return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
  elif metric == 'manhattan':
      # Manhattan distance (sum of absolute differences)
      return np.sum(np.abs(a - b))
  elif metric == 'chebyshev':
      # Chebyshev distance (maximum absolute difference)
      return np.max(np.abs(a - b))
  else:
      raise ValueError(f"Unsupported metric: {metric}")


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

def generate_sample(sample_id, min_value, max_value, vector_size, metric='euclidean'):
    sample_pair = generate_vector_pair(min_value, max_value, vector_size)
    sample_distance = calculate_distance(sample_pair[0], sample_pair[1], metric)
    return sample_pair, sample_distance

def generate_sample_data_with_multithreading(number_of_samples, min_value, max_value, vector_size, metric='euclidean'):
    sample_pairs = []
    sample_distances = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(generate_sample, i, min_value, max_value, vector_size, metric) for i in
                   range(number_of_samples)]

        completed_count = 0
        for future in concurrent.futures.as_completed(futures):
            completed_count += 1
            if completed_count % 100 == 0:  # Print progress every 100 samples
                print(f"\r{completed_count}/{number_of_samples} samples generated",end="")

            sample_pair, sample_distance = future.result()
            sample_pairs.append(sample_pair)
            sample_distances.append(sample_distance)
    print();
    return np.array(sample_pairs), np.array(sample_distances)




# tests
if __name__ == '__main__':
    x_data, y_data = generate_sample_data_for_recurrent(10, 0, 1, 2)
    for line1, line2 in zip(x_data, y_data):
        print(line1, line2)

