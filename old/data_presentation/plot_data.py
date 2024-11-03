import matplotlib.pyplot as plt
import pandas as pd


def plot_recurrent_layers():
    FILE_PATH = "../saved_results/old/important/searching_fc_layers.csv"
    df = pd.read_csv(FILE_PATH)

    fig, ax = plt.subplots(5, 1, figsize=(10, 10))
    plot_index = 0

    df.drop(["Multiplier", "Lr", "Layers r", "Loss", "Validation loss"], axis=1, inplace=True)
    for input_dim in df["Input dim"].unique():
        df_id = df[df["Input dim"] == input_dim]
        for hidden_dim in df_id["Hidden dim"].unique():
            df_hid = df_id[df_id["Hidden dim"] == hidden_dim]
            means = df_hid.groupby(["Layers fc"])["Epochs"].mean()
            ax[plot_index].plot(df_hid["Layers fc"].unique(), means)
            ax[plot_index].set_xlabel("Number of fully layers in output module")
            ax[plot_index].set_ylabel("Number of epochs")
            # ax[plot_index].set_yscale("log")
        ax[plot_index].set_title(f"Number of fully connected layers in output module (input_dim = {input_dim})")
        ax[plot_index].legend([f"Hidden dim = {hd}" for hd in df_id["Hidden dim"].unique()])
        ax[plot_index].grid(True)
        plot_index += 1

    plt.tight_layout()
    plt.show()


def plot_hidden_dim_r():
    FILE_PATH = "../saved_results/old/important/searching_hidden_dim_fc.csv"
    df = pd.read_csv(FILE_PATH)

    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    plot_index = 0

    df.drop(["Multiplier", "Lr", "Layers r", "Loss", "Validation loss", "Hidden dim r", "Layers r"], axis=1, inplace=True)
    for input_dim in df["Input dim"].unique():
        df_id = df[df["Input dim"] == input_dim]
        means = df_id.groupby(["Hidden dim fc"])["Epochs"].mean()
        ax[plot_index].plot(df_id["Hidden dim fc"].unique(), means)
        ax[plot_index].set_xlabel("Number of neurons in layer of output module")
        ax[plot_index].set_ylabel("Number of epochs")
        # ax[plot_index].set_yscale("log")
        ax[plot_index].set_title(f"Number of neurons in layer in output module (input_dim = {input_dim})")
        # ax[plot_index].legend([f"Hidden dim = {hd}" for hd in df_id["Hidden dim"].unique()])
        ax[plot_index].grid(True)
        plot_index += 1

    plt.tight_layout()
    plt.show()


def plot_data_demand():
    FILE_PATH = "../saved_results/old/data_demand_recurrent.csv"
    df = pd.read_csv(FILE_PATH)
    

    df.drop(["Multiplier", "Lr", "Layers r", "Loss", "Validation loss", "Hidden dim r", "Hidden dim fc", "Layers fc"], axis=1, inplace=True)
    means = df.groupby(["Input dim"])["Epochs"].mean()
    plt.plot(df["Input dim"].unique(), means)
    plt.title("Data demand for recurrent model")
    plt.xlabel("Input dim")
    plt.ylabel("Number of epochs [epoch = 64 data points]")
    plt.show()


if __name__ == "__main__":
    plot_data_demand()