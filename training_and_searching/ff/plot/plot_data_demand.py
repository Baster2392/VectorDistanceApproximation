import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = '../data_demand_and_complexity_results/results100_1000.csv'
data = pd.read_csv(file_path)

sns.set(style="whitegrid")

plt.figure(figsize=(12, 8))
sns.lineplot(data=data, x="Dataset size", y="Test Loss", hue="Input dimension", marker="o")

plt.xlabel("Dataset size")
plt.ylabel("Test Loss")
plt.title("Test Loss vs Dataset size for Different Input Dimensions")

plt.legend(title="Input Dimension")
plt.savefig('./LossVsDatasetSize.png')

target_test_loss = 0.15
plt.figure(figsize=(12, 8))

filtered_data = (
    data[data['Test Loss'] < target_test_loss]
    .sort_values(by='Test Loss', ascending=False)
    .drop_duplicates(subset=['Input dimension'])
)

sns.lineplot(data=filtered_data, x="Input dimension", y="Dataset size")
plt.xlabel("Input Dimension")
plt.ylabel("Dataset size")
plt.title(f"Dataset size required for Test Loss = {target_test_loss}")
plt.savefig('./requiredDatasetSize.png')
