import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.use('TkAgg')

data = pd.read_csv('training_results_5_15.csv')

# Grouping by 'n' and 'num_bits' to calculate the mean test_loss
grouped_data = data.groupby(['n', 'num_bits'])['test_loss'].mean().reset_index()

# Unique values of 'n' for grouping
unique_n = sorted(grouped_data['n'].unique())

# Create subplots for each n
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
fig.suptitle('Zależność test_loss od num_bits dla różnych wartości n', fontsize=16)

# Plot for each value of n in separate subplots
for i, n_value in enumerate(unique_n):
    subset = grouped_data[grouped_data['n'] == n_value]
    ax = axes[i]
    sns.lineplot(data=subset, x='num_bits', y='test_loss', marker='o', ax=ax)
    ax.set_title(f'n = {n_value}')
    ax.set_xlabel('num_bits')
    if i == 0:
        ax.set_ylabel('test_loss')
    else:
        ax.set_ylabel('')

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
