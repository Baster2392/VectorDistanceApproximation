import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors


"""
This is a somewhat powerful data visualisation tool, that allows to plot the data after grouping it, 
filtering it and transforming it.

Initially I meant the generate_plots_table function for our usage primarily, since at the time ,
I believed the resulting graphs were somewhat unreadable, however informative.
As of right now i wholeheartedly believe that such table of plots can be used in our paper.

The usage of the functions are explained in them, 
also the main block contains many example usages to show what can be done with this tool.


I plan on providing a list as a filter_criteria, so that we could for example merge data from different number of labels.
"""


def plot_csv_columns(csv_file, x_columns, y_column, x_combine_method='sum', filter_columns=None, filter_criteria=None,
                     merge_column=None, xlabel=None, ylabel=None, title=None, plot_type='scatter'):
    # Read the CSV file into a DataFrame
    data = pd.read_csv(csv_file)

    # Apply filtering for filter columns if specified
    if filter_columns and filter_criteria:
        for filter_column, criterion in zip(filter_columns, filter_criteria):
            filter_value, filter_range = criterion
            if filter_value is not None:
                data = data[data[filter_column] == filter_value]
            elif filter_range is not None:
                data = data[(data[filter_column] >= filter_range[0]) & (data[filter_column] <= filter_range[1])]

    # Merge rows based on the merge_column
    if merge_column:
        numeric_columns = data.select_dtypes(include=np.number).columns.tolist()
        numeric_columns.remove(merge_column)
        grouped_numeric = data.groupby(merge_column)[numeric_columns].mean().reset_index()

        non_numeric_columns = data.select_dtypes(exclude=np.number).columns.tolist()
        for col in non_numeric_columns:
            if col != merge_column:
                grouped_numeric[col] = ''

        data = grouped_numeric

    # Combine the values of x-columns based on the specified method
    if x_combine_method == 'sum':
        x_data = data[x_columns].sum(axis=1)
    elif x_combine_method == 'mean':
        x_data = data[x_columns].mean(axis=1)
    elif x_combine_method == 'min':
        x_data = data[x_columns].min(axis=1)
    elif x_combine_method == 'capacity':
        hidden_dim = data['Hidden Dimension']
        siamese_layers = data['Siamese Layers']
        shared_layers = data['Shared Layers']
        x_data = 2 * (hidden_dim ** siamese_layers) + (2 * hidden_dim) ** shared_layers
    elif x_combine_method == 'max':
        x_data = data[x_columns].max(axis=1)
    elif x_combine_method == 'difference':
        if len(x_columns) == 2:
            x_data = data[x_columns[0]] - data[x_columns[1]]
        else:
            raise ValueError("Difference method requires exactly two x-columns.")
    else:
        raise ValueError("Invalid x_combine_method. Choose from 'sum', 'mean', 'min', 'max', 'difference'.")

    y_data = data[y_column]

    # Plot the data
    if plot_type == 'scatter':
        plt.scatter(x_data, y_data)
    elif plot_type == 'line':
        plt.plot(x_data, y_data)
    elif plot_type == 'bar':
        plt.bar(x_data, y_data)
    else:
        raise ValueError("Invalid plot_type. Choose from 'scatter', 'line', 'bar'.")

    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if title:
        plt.title(title)
    else:
        plt.title(f'{y_column} vs {x_combine_method.capitalize()} of {"+".join(x_columns)}')
    plt.show()

def generate_plots_table(csv_file, x_columns, y_column, x_combine_method='sum', filter_column_1=None, filter_values_1=range(1, 6), filter_column_2=None, filter_values_2=range(1, 6), merge_column=None):
    data = pd.read_csv(csv_file)

    # Determine the size of the plot grid
    num_values_1 = len(filter_values_1)
    num_values_2 = len(filter_values_2)

    fig, axes = plt.subplots(num_values_1, num_values_2, figsize=(15, 10))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    fig.suptitle(f'{filter_column_2}', fontsize=16, y=0.95, fontweight='bold')
    fig.text(0.05, 0.6, f'{filter_column_1}', ha='right', va='center', fontsize=16, fontweight='bold',
            rotation=90)  # Adjust position and rotation

    # Find the min and max of the smallest y-values among all plots
    min_values = []
    for value_1 in filter_values_1:
        for value_2 in filter_values_2:
            filtered_data = data[(data[filter_column_1] == value_1) & (data[filter_column_2] == value_2)]
            if not filtered_data.empty:
                min_values.append(filtered_data[y_column].min())
    global_min = min(min_values)
    global_max = max(min_values)

    norm = mcolors.Normalize(vmin=global_min, vmax=global_max)
    cmap = plt.get_cmap('coolwarm')

    for i, value_1 in enumerate(filter_values_1):
        for j, value_2 in enumerate(filter_values_2):
            ax = axes[i, j]

            # Filter data for the current combination of filter values
            filtered_data = data[(data[filter_column_1] == value_1) & (data[filter_column_2] == value_2)]

            if filtered_data.empty:
                ax.set_visible(False)
                continue

            # Merge rows based on the merge_column after filtering
            if merge_column and merge_column in filtered_data.columns:
                numeric_columns = filtered_data.select_dtypes(include=np.number).columns.tolist()
                if merge_column in numeric_columns:
                    numeric_columns.remove(merge_column)
                grouped_numeric = filtered_data.groupby(merge_column)[numeric_columns].mean().reset_index()

                non_numeric_columns = filtered_data.select_dtypes(exclude=np.number).columns.tolist()
                for col in non_numeric_columns:
                    if col != merge_column:
                        grouped_numeric[col] = ''

                filtered_data = grouped_numeric

            # Combine the values of x-columns based on the specified method
            if x_combine_method == 'sum':
                x_data = filtered_data[x_columns].sum(axis=1)
            elif x_combine_method == 'mean':
                x_data = filtered_data[x_columns].mean(axis=1)
            elif x_combine_method == 'min':
                x_data = filtered_data[x_columns].min(axis=1)
            elif x_combine_method == 'max':
                x_data = filtered_data[x_columns].max(axis=1)
            elif x_combine_method == 'difference':
                if len(x_columns) == 2:
                    x_data = filtered_data[x_columns[0]] - filtered_data[x_columns[1]]
                else:
                    raise ValueError("Difference method requires exactly two x-columns.")
            else:
                raise ValueError("Invalid x_combine_method. Choose from 'sum', 'mean', 'min', 'max', 'difference'.")

            y_data = filtered_data[y_column]

            # Determine the minimum y value for the current plot and apply colorscale
            min_y = y_data.min()
            color = cmap(norm(min_y))
            ax.set_facecolor(color)

            # Plot the data on the current subplot
            ax.scatter(x_data, y_data)
            ax.set_xlabel(f'{x_combine_method.capitalize()} of {"+".join(x_columns)}')
            ax.set_ylabel(y_column)
            ax.set_title(f'{value_1}, {value_2}')

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=axes.ravel().tolist(), orientation='horizontal', label=f'{y_column} Colorbar')

    plt.show()


if __name__ == '__main__':
    current_dir = os.getcwd()
    print(f"Current Working Directory: {current_dir}")
    df = pd.read_csv('../saved_results/results.csv')
    csv_file = '../saved_results/results.csv'

    plot_csv_columns(csv_file, ['Input Dim', 'Hidden Dimension'], 'Test Loss',
                    merge_column='Hidden Dimension')
    plot_csv_columns(csv_file, ['Input Dim', 'Hidden Dimension'], 'Test Loss',
                     )

    # Example usage:
    generate_plots_table(csv_file, ['Hidden Dimension'], 'Test Loss', x_combine_method='sum', filter_column_1='Shared Layers', filter_values_1=range(1, 8), filter_column_2='Siamese Layers', filter_values_2=range(1, 8))
    plot_csv_columns(csv_file, ['Hidden Dimension', 'Siamese Layers', 'Shared Layers'], 'Test Loss',
                     x_combine_method='capacity',
                     merge_column='Hidden Dimension')
