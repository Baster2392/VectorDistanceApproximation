import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use('TkAgg')

n = 5
hidden_dim = 32
hidden_layers = 3
epochs = 100
granulation = 5
loss_tolerance = 0.0
granulation_mode = False
epsilon=0.0

data = pd.read_csv(f'./progress_results/{n}_{hidden_dim}_{hidden_layers}_{epochs}_{granulation}_{loss_tolerance}_{granulation_mode}_{epsilon}.csv')

plt.plot(data['epoch'], data['train_loss'], label='Train loss')
plt.plot(data['epoch'], data['val_loss'], label='Val loss')
plt.plot(data['epoch'], data['val_normal_loss'], label='Val loss (normal distribution)')
plt.yscale('log')
plt.xlabel('Loss value')
plt.ylabel('Test Loss (log scale)')
plt.title(f'Test Loss in each epoch for n={n}, granulation={granulation}, epsilon={epsilon}')
plt.legend()
plt.show()
