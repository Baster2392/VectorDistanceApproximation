import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = 'training_and_searching/transformer/data_demand/results/data_demand1.csv'
data = pd.read_csv(file_path)

sns.set(style="whitegrid")

plt.figure(figsize=(12, 8))
sns.lineplot(data=data, x="Train Dataset Size", y="Test Loss", hue="Input Dimensionality", marker="o")

plt.xlabel("Train Dataset size")
plt.ylabel("Test Loss")
plt.title("Test Loss vs Dataset size for Different Input Dimensions")

plt.legend(title="Input Dimension")
plt.savefig('training_and_searching/plot/plots/LossVsDatasetSizeTransformer.png')

target_test_loss = 0.10
plt.figure(figsize=(12, 8))

filtered_data = (
    data[data['Test Loss'] < target_test_loss]
    .sort_values(by='Test Loss', ascending=False)
    .drop_duplicates(subset=['Input Dimensionality'])
)

sns.lineplot(data=filtered_data, x="Input Dimensionality", y="Test Dataset Size")
plt.xlabel("Input Dimension")
plt.ylabel("Dataset size")
plt.title(f"Dataset size required for Test Loss = {target_test_loss}")
plt.savefig('training_and_searching/plot/plots/requiredDatasetSize.png')
