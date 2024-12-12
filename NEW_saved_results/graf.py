import pandas as pd
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

# Definicja kolorów i nazw dla różnych rozmiarów 'Input_dim'
files = [
    'C:\\Users\\Radek\\PycharmProjects\\VectorDistanceApproximation\\NEW_saved_results\\NEW_data_demand_recurrent_SmoothL1Loss_AdamW_64_500_const_epoch_small.csv'
]

color_map = plt.cm.get_cmap('viridis')  # Możesz zmienić na 'plasma', 'rainbow' itp.
for file in files:
    # Wczytanie danych z pliku CSV
    df = pd.read_csv(file)
    print(df.head())  # Wyświetlenie pierwszych kilku wierszy, aby sprawdzić dane

    # Iteracja po unikalnych wartościach 'Input_dim' i tworzenie linii dla każdej wartości
    counter = 0
    for input_dim in df['Input_dim'].unique():
        df_filtered = df[df['Input_dim'] == input_dim]  # Filtrowanie dla danego input_dim

        # Grupowanie i wybór średnich wartości dla kolumny 'Data_demand'
        df_avg = df_filtered.groupby('Data_demand')['Test_loss'].mean().reset_index()

        x = df_avg['Data_demand']
        y = df_avg['Test_loss']

        # Tworzenie wykresu dla każdego zbioru danych
        color = color_map(counter)  # Counter powinien być w zakresie 0-1
        plt.plot(x, y, marker='o', linestyle='-', color=color, label=f'Input dim {input_dim}')
        counter += 1/len(df['Input_dim'].unique())

# Dodanie tytułu, etykiet i legendy
plt.title('Średni test loss dla różnych Input Dim')
plt.xlabel('Data Demand')
plt.ylabel('loss')
plt.grid(True)
plt.legend(title='Input Dim')  # Dodanie legendy z tytułem
#plt.yscale('log', base=10)  # Logarytm dwójkowy

# Wyświetlenie wykresu
plt.show()
