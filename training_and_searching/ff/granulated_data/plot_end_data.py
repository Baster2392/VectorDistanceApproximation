import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use('TkAgg')

data = pd.read_csv('saved/end_results/training_results_with_normal_distribution.csv')

# Group data by 'n' for plotting
unique_n = data['n'].unique()

# Create a figure
plt.figure(figsize=(14, 8))

for i, n_val in enumerate(unique_n, start=1):
    subset = data[data['n'] == n_val]
    subset = subset[subset['granulation_mode'] == False]
    granulation = subset['granulation']
    test_loss = subset['test_loss']
    test_normal_loss = subset['test_normal_loss']

    # Plot test_loss
    plt.subplot(2, 1, 1)
    plt.plot(granulation, test_loss, marker='o', label=f'n={n_val}')
    plt.yscale('log')
    plt.xlabel('Granulation')
    plt.ylabel('Test Loss (log scale)')
    plt.title('Test Loss vs Granulation')
    plt.legend()

    # Plot test_normal_loss
    plt.subplot(2, 1, 2)
    plt.plot(granulation, test_normal_loss, marker='o', label=f'n={n_val}')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Granulation (log scale)')
    plt.ylabel('Test Normal Loss (log scale)')
    plt.title('Test Normal Loss vs Granulation (Log Scale)')
    plt.legend()

plt.tight_layout()
plt.show()
