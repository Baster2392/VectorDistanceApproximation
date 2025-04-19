input_dim = 100
i = 10
layers_config = [128 * i, 100 * i, 64 * i, 50 * i]

# Compute Theoretical Computational Complexity
input_dim = input_dim
layers_config = layers_config
shared_complexity = 0
in_dim = input_dim
for num_neurons in layers_config[:-1]:
    shared_complexity += in_dim * num_neurons
    in_dim = num_neurons
shared_complexity += in_dim * layers_config[-1]
theoretical_complexity = 2 * shared_complexity + layers_config[-1] * layers_config[-1] + layers_config[-1] * 1

print(theoretical_complexity)