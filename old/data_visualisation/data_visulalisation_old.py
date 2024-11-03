import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""
Legacy versions of the functions in the data_visualisation.py file. I advise against using them, 
i decided to keep them just in case the current code malfunctions - in that case however it would be better to
improve the current code rather than use these. Treat them as last resort.
"""

def plot_combined_plots(df, first_metric='Last Min Loss', second_metric=None, language='en'):
    if second_metric is None:
        metric = first_metric
    else:
        metric = f'{first_metric}-{second_metric}'
        df[metric] = df[first_metric] - df[second_metric]

    sns.set_palette("husl")
    fig, axs = plt.subplots(len(df['Siamese Layers'].unique()), len(df['Shared Layers'].unique()), figsize=(20, 20))

    smallest_losses = [group_df[metric].min() for _, group_df in df.groupby(['Siamese Layers', 'Shared Layers'])]
    levels = np.linspace(min(smallest_losses), max(smallest_losses), 256)
    colors = [(0, 1, 0), (1, 0, 0)]
    cmap = plt.cm.colors.LinearSegmentedColormap.from_list('custom_colormap', colors, N=len(levels))

    for idx, ((num_siamese_layers, num_shared_layers), group_df) in enumerate(
            df.groupby(['Siamese Layers', 'Shared Layers'])):
        row = idx // len(df['Shared Layers'].unique())
        col = idx % len(df['Shared Layers'].unique())
        ax = axs[row, col]

        smallest_loss = group_df[metric].min()
        sorted_df = group_df.sort_values(by='Hidden Dimension')
        ax.plot(sorted_df['Hidden Dimension'], sorted_df[metric], color='blue', marker='o')

        ax.set_title(
            f'Siamese: {num_siamese_layers}, Shared: {num_shared_layers}' if language == 'en' else f'Syjamskie: {num_siamese_layers}, Wspólne: {num_shared_layers}')
        ax.set_xlabel('Hidden Dimension' if language == 'en' else 'Wymiar Ukryty')
        ax.set_ylabel(metric)
        ax.grid(True)

        normalized_loss = (smallest_loss - min(smallest_losses)) / (max(smallest_losses) - min(smallest_losses))
        ax.set_facecolor(cmap(normalized_loss))

    plt.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.15, -0.05, 0.7, 0.05])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(smallest_losses), vmax=max(smallest_losses)))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(metric)

    plt.tight_layout()
    plt.savefig('images/combined_plots.png', bbox_inches='tight')
    plt.show()
def plot_min_loss_vs_shared_layers(df, first_metric='Last Min Loss', second_metric=None, language='en'):
    if second_metric is None:
        metric = first_metric
    else:
        metric = f'{first_metric}-{second_metric}'
        df[metric] = df[first_metric] - df[second_metric]

    min_loss_shared = df.groupby('Shared Layers')[metric].min().reset_index()
    min_hidden_shared = df.merge(min_loss_shared, on=['Shared Layers', metric], how='inner')['Hidden Dimension']
    avg_losses_shared = df[df['Hidden Dimension'].isin(min_hidden_shared)].groupby('Shared Layers')[
        metric].mean().reset_index()

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=avg_losses_shared, x='Shared Layers', y=metric, marker='o')
    plt.xlabel("Shared Layers" if language == 'en' else "Wspólne Warstwy")
    plt.ylabel(f"Average {metric}" if language == 'en' else f"Średnia {metric}")
    plt.title(
        f"Average {metric} found after 1000 epoch vs Number of Shared Layers" if language == 'en' else f"Średnia {metric} po 1000 epokach w zależności od liczby wspólnych warstw")
    plt.grid(True)
    plt.show()
def plot_min_loss_vs_siamese_layers(df, first_metric='Last Min Loss', second_metric=None, language='en'):
    if second_metric is None:
        metric = first_metric
    else:
        metric = f'{first_metric}-{second_metric}'
        df[metric] = df[first_metric] - df[second_metric]

    min_loss_siamese = df.groupby('Siamese Layers')[metric].min().reset_index()
    min_hidden_siamese = df.merge(min_loss_siamese, on=['Siamese Layers', metric], how='inner')['Hidden Dimension']
    avg_losses_siamese = df[df['Hidden Dimension'].isin(min_hidden_siamese)].groupby('Siamese Layers')[
        metric].mean().reset_index()

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=avg_losses_siamese, x='Siamese Layers', y=metric, marker='o')
    plt.xlabel("Siamese Layers" if language == 'en' else "Warstwy Syjamskie")
    plt.ylabel(f"Average {metric}" if language == 'en' else f"Średnia {metric}")
    plt.title(
        f"Average {metric} found after 1000 epoch vs Number of Siamese Layers" if language == 'en' else f"Średnia {metric} po 1000 epokach w zależności od liczby warstw syjamskich")
    plt.grid(True)
    plt.show()
def plot_min_loss_vs_hidden_dim(df, first_metric='Last Min Loss', second_metric=None, language='en'):
    if second_metric is None:
        metric = first_metric
    else:
        metric = f'{first_metric}-{second_metric}'
        df[metric] = df[first_metric] - df[second_metric]

    avg_losses = df.groupby('Hidden Dimension')[metric].mean().reset_index()

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=avg_losses, x='Hidden Dimension', y=metric, marker='o')
    plt.xlabel("Hidden Dimension" if language == 'en' else "Wymiar Ukryty")
    plt.ylabel(f"Average {metric}" if language == 'en' else f"Średnia {metric}")
    plt.title(
        f"Average {metric} found after 1000 epoch vs Number of Neurons" if language == 'en' else f"Średnia {metric} po 1000 epokach w zależności od liczby neuronów")
    plt.grid(True)
    plt.show()
def plot_siamese_vs_shared_trends(df, first_metric='Last Min Loss', second_metric=None, language='en'):
    if second_metric is None:
        metric = first_metric
    else:
        metric = f'{first_metric}-{second_metric}'
        df[metric] = df[first_metric] - df[second_metric]

    siamese_trend = df.groupby('Siamese Layers')[metric].mean().reset_index()
    shared_trend = df.groupby('Shared Layers')[metric].mean().reset_index()

    fig, ax = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    sns.lineplot(data=siamese_trend, x='Siamese Layers', y=metric, marker='o', ax=ax[0])
    ax[0].set_title(
        f'Average {metric} vs Siamese Layers' if language == 'en' else f'Średnia {metric} vs Warstwy Syjamskie')
    ax[0].set_xlabel('Siamese Layers' if language == 'en' else 'Warstwy Syjamskie')
    ax[0].set_ylabel(f'Average {metric}' if language == 'en' else f'Średnia {metric}')
    ax[0].grid(True)

    sns.lineplot(data=shared_trend, x='Shared Layers', y=metric, marker='o', ax=ax[1])
    ax[1].set_title(
        f'Average {metric} vs Shared Layers' if language == 'en' else f'Średnia {metric} vs Warstwy Wspólne')
    ax[1].set_xlabel('Shared Layers' if language == 'en' else 'Warstwy Wspólne')
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()
def plot_min_loss_hd75(df, first_metric='Last Min Loss', second_metric=None, language='en'):
    if second_metric is None:
        metric = first_metric
    else:
        metric = f'{first_metric}-{second_metric}'
        df[metric] = df[first_metric] - df[second_metric]

    df_hd100 = df[df['Hidden Dimension'] == 75]
    avg_min_loss_siamese = df_hd100.groupby('Siamese Layers')[metric].mean().reset_index()
    avg_min_loss_shared = df_hd100.groupby('Shared Layers')[metric].mean().reset_index()

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    sns.lineplot(data=avg_min_loss_siamese, x='Siamese Layers', y=metric, marker='o', ax=axs[0])
    axs[0].set_title(
        f'Minimum Loss vs Siamese Layers (hidden_dim=75)' if language == 'en' else f'Minimalna Strata vs Warstwy Syjamskie (hidden_dim=75)')
    axs[0].set_xlabel('Siamese Layers' if language == 'en' else 'Warstwy Syjamskie')
    axs[0].set_ylabel('Minimum Loss' if language == 'en' else 'Minimalna Strata')
    axs[0].grid(True)

    sns.lineplot(data=avg_min_loss_shared, x='Shared Layers', y=metric, marker='o', ax=axs[1])
    axs[1].set_title(
        f'Minimum Loss vs Shared Layers (hidden_dim=75)' if language == 'en' else f'Minimalna Strata vs Warstwy Wspólne (hidden_dim=75)')
    axs[1].set_xlabel('Shared Layers' if language == 'en' else 'Warstwy Wspólne')
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()