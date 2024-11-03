import numpy as np
import matplotlib.pyplot as plt
import torch
import old.data_generators.vector_generator as vg
from old.models.recurrect_model import LSTMModel

final_model = LSTMModel(100, 64, 300, 2, 3)
final_model.load_state_dict(torch.load("../saved_models/100_recurrent_1719507480.6065662.pth"))

X_test, Y_test = vg.generate_sample_data_for_recurrent(10000, 0, 1, 100, True)
X_test, Y_test = torch.tensor(X_test, dtype=torch.float), torch.tensor(Y_test, dtype=torch.float)

# Convert predictions and test targets back to NumPy arrays for plotting
predictions = final_model(X_test).detach().cpu().numpy().squeeze()
y_test = Y_test.cpu().numpy().squeeze()

# Calculate error statistics
errors = predictions - y_test
relative_errors = (predictions - y_test) / y_test
median_relative_error = np.median(abs(relative_errors))
mean_relative_error = np.mean(abs(relative_errors))
max_relative_error = np.max(relative_errors)
min_relative_error = np.min(relative_errors)
median_error = np.median(errors)
mean_error = np.mean(errors)
max_overestimation = np.max(errors)
max_underestimation = np.min(errors)

# Create the plot for predictions vs. true values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions, color='blue', label='Predictions')

# Upper error line (5% above true values)
upper_error = y_test * 1.05
#plt.plot(y_test, upper_error, color='green', linestyle='--', label='5% Overestimation')

# Lower error line (5% below true values)
lower_error = y_test * 0.95
#plt.plot(y_test, lower_error, color='green', linestyle='--', label='5% Underestimation')

# Add reference line for true values (unchanged)
plt.plot(y_test, y_test, color='red', linestyle='--', label='Exact estimation')

plt.text(0.8, 0.1, f'Median Error: {median_relative_error:.4f}', transform=plt.gca().transAxes)
plt.text(0.8, 0.05, f'Mean Error: {mean_relative_error:.4f}', transform=plt.gca().transAxes)

# Add max overestimation line
plt.plot(y_test, y_test + max_overestimation, color='magenta', linestyle='--', label=f'Max Overestimation {max_relative_error:.4f}')

# Add max underestimation line
plt.plot(y_test, y_test + max_underestimation, color='cyan', linestyle='--', label=f'Max Underestimation {min_relative_error:.4f}')

plt.xlabel('True Euclidean Distance')
plt.ylabel('Predicted Euclidean Distance')
plt.title('Predictions vs. True Values with Error Lines')
plt.legend()
plt.grid(True)

# Display the plot
plt.show()

# Plot the first 50 true values and predictions
plt.figure(figsize=(8, 6))
plt.plot(y_test[:50], label='True Values')
plt.plot(predictions[:50], label='Predictions')
plt.legend()
plt.xlabel('Sample Index')
plt.ylabel('Euclidean Distance')
plt.title('First 50 True Values and Predictions')
plt.grid(True)
plt.show()
abs_relative_errors = relative_errors


def calculate_range_frequency(errors, start, end):
    """ Calculate the frequency of errors within a specific range. """
    count = np.sum((errors >= start) & (errors <= end))
    total = len(errors)
    return count / total * 100  # Convert to percentage


# Define error ranges and calculate frequencies
error_ranges = [(-0.005, 0.005), (-0.01, 0.01), (-0.015, 0.015)]
range_frequencies = {f'{start} to {end}': calculate_range_frequency(abs_relative_errors, start, end) for start, end in error_ranges}


# Plot the error distribution
num_bins = 100
plt.figure(figsize=(8, 6))
plt.hist(abs_relative_errors, bins=num_bins, color='blue', edgecolor='black', density=True)

# Convert frequencies to percentages
bin_width = (max(abs_relative_errors) - min(abs_relative_errors)) / num_bins
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y * bin_width * 100:.1f}%'))

from matplotlib.ticker import FuncFormatter


# Function to format tick labels as percentages
def to_percent(y, position):
    s = f"{100 * y:.1f}%"

    if plt.rcParams['text.usetex']:
        return s + r'$\%$'
    else:
        return s


plt.gca().xaxis.set_major_formatter(FuncFormatter(to_percent))


# Add text for the frequencies of specific ranges
for range, frequency in range_frequencies.items():
    plt.text(0.95, 0.95 - 0.05 * list(range_frequencies.keys()).index(range),
             f'Error {range} frequency: {frequency:.2f}%',
             transform=plt.gca().transAxes, horizontalalignment='right')


plt.xlabel('Prediction Error')
plt.ylabel('Frequency (%)')
plt.title('Error Distribution')
plt.grid(True)
plt.show()
