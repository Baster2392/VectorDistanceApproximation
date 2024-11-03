import matplotlib.pyplot as plt
import pandas as pd


def plot_grid_search_recurrent_layers_r():
    df = pd.read_csv('../saved_results/old/important/normalized_grid_search_recurrent_layers_32_64.csv')

    # Grupowanie danych względem 'Hidden dim' i 'Layers', obliczanie średniej wartości 'Epochs'
    grouped = df.groupby(['Hidden dim', 'Layers'])['Epochs'].mean().reset_index()

    # Tworzenie dwóch wykresów - jeden dla 'Hidden dim' = 32, drugi dla 'Hidden dim' = 64
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))

    hidden_dims = [32, 64]
    for i, hidden_dim in enumerate(hidden_dims):
        subset = grouped[grouped['Hidden dim'] == hidden_dim]
        axs[i].plot(subset['Layers'], subset['Epochs'])
        axs[i].set_title(f'Layers in recurrent module (Hidden dim = {hidden_dim})')
        axs[i].set_xlabel('Number of layers in recurrent module')
        axs[i].set_ylabel('Average Epochs')
        axs[i].grid(True)

    plt.tight_layout()
    plt.show()


def plot_grid_search_recurrent_layers_fc():
    df = pd.read_csv(
        '../saved_results/old/important/normalized_grid_search_recurrent_fc_search_validation_20_40_80.csv')
    df = df.drop("Hidden dim", axis=1)
    df = df.groupby(["Layers"])['Epochs'].mean().reset_index()

    plt.plot(df["Layers"], df["Epochs"], marker='o')
    plt.title("Number of fc layers")
    plt.xlabel("Number of layers")
    plt.ylabel("Average Epochs (Normalized)")

    plt.show()


if __name__ == "__main__":
    plot_grid_search_recurrent_layers_r()

