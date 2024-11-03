import numpy as np
import pandas as pd


def normalize_grid_search_recurrent_layers_r():
    df = pd.read_csv('../saved_results/old/grid_search_recurrent_layers_32_64.csv')

    df.drop(["Multiplier", "Lr", "Loss"], axis=1, inplace=True)
    df_input_dims = df["Input dim"].unique()
    new_df = []

    for input_dim in df_input_dims:
        df_id = df[df["Input dim"] == input_dim]
        df_hidden_dims = df_id["Hidden dim"].unique()
        for hidden_dim in df_hidden_dims:
            df_hd = df_id[df_id["Hidden dim"] == hidden_dim]
            min_epochs = np.min(df_hd["Epochs"])
            for index, row in df_hd.iterrows():
                new_row = {
                    "Input dim": input_dim,
                    "Hidden dim": hidden_dim,
                    "Layers": row["Layers"],
                    "Epochs": float(row["Epochs"]) / float(min_epochs),
                }
                new_df.append(new_row)

    df = df.groupby(["Layers"])["Epochs"].mean()
    new_df = pd.DataFrame(new_df)
    new_df.to_csv("../saved_results/normalized_grid_search_recurrent_layers_32_64.csv", index=False)
    print(new_df)


def normalize_grid_search_recurrent_layers_fc():
    df = pd.read_csv('../saved_results/old/grid_search_recurrent_fc_search_validation_20_40_80.csv')

    df.drop(["Multiplier", "Lr", "Loss", "Hidden dim", "Layers r"], axis=1, inplace=True)
    df_input_dims = df["Input dim"].unique()
    new_df = []

    for input_dim in df_input_dims:
        df_id = df[df["Input dim"] == input_dim]
        df_num_layers = df_id["Layers fc"].unique()
        min_epochs = np.min(df_id["Epochs"])
        for num_layers in df_num_layers:
            df_nl = df_id[df_id["Layers fc"] == num_layers]
            for index, row in df_nl.iterrows():
                new_row = {
                    "Input dim": input_dim,
                    "Hidden dim": 32,
                    "Layers": row["Layers fc"],
                    "Epochs": float(row["Epochs"]) / float(min_epochs),
                }
                new_df.append(new_row)

    df = df.groupby(["Layers fc"])["Epochs"].mean()
    new_df = pd.DataFrame(new_df)
    new_df.to_csv("../saved_results/important/normalized_grid_search_recurrent_fc_search_validation_20_40_80.csv", index=False)
    print(new_df)


if __name__ == "__main__":
    normalize_grid_search_recurrent_layers_fc()
