import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = 'D:\\Studia\\Sem 4\\SI\\Projekt\\VectorDistanceCalculator\\training_and_searching\\ff\\data_demand_and_complexity_results\\results_100_layers.csv'
data = pd.read_csv(file_path)

data = data[['Input dimension', 'Number of Layers', 'Test Loss']]
data = data.groupby(['Input dimension', 'Number of Layers']).mean()
print(data.head())

sns.set(style="whitegrid")

plt.figure(figsize=(12, 8))
sns.lineplot(data=data, x="Number of Layers", y="Test Loss", hue="Input dimension", marker="o")

plt.xlabel("Number of layers")
plt.ylabel("Test Loss")
plt.title("Test Loss vs Number of Layers for Different Input Dimensions")

plt.legend(title="Input Dimension")
plt.show()
