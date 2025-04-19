import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Wczytanie danych
file_path = "D:\\Studia\\Sem 4\\SI\\Projekt\VectorDistanceCalculator\\training_and_searching\\plot\\additional_data\\data_demand_100_comparison.csv"
df = pd.read_csv(file_path)

# Usunięcie zbędnych spacji w nazwach kolumn
df.columns = df.columns.str.strip()

# Ustawienie stylu wykresów
sns.set(style="whitegrid")

# Unikalne architektury sieci neuronowych
architectures = df["Architecture"].unique()

# Tworzenie wykresów
plt.figure(figsize=(12, 6))
for arch in architectures:
    subset = df[df["Architecture"] == arch]
    plt.plot(subset["Train Dataset Size"], subset["Test Loss"], marker='o', label=arch)

# Konfiguracja wykresu
plt.xlabel("Train Dataset Size")
plt.ylabel("Test Loss")
plt.title("Test Loss vs Train Dataset Size for Different Architectures")
plt.legend(title="Architecture")
plt.show()
