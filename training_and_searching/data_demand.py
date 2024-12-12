import math
import colorsys
import numpy as np
import torch
from torch import nn
import time
import os
from torch.utils.data import DataLoader, TensorDataset


# Radkowe rzeczy
import tkinter as tk
from tkinter import messagebox

root = tk.Tk()
root.withdraw()

def show_alert(dane):
    alert = tk.Toplevel()
    alert.configure(bg="lightblue")
    alert.relief = tk.RAISED
    alert.title("To by nic nie dało")
    alert.geometry("500x200")

    label = tk.Label(alert, text=f"Brak progresu od 50 000 epoch dla danych:\n {dane}",font=("Comic Sans MS", 15, "bold italic"),bg="lightblue", fg="darkblue")

    label.pack(pady=20)

    alert.after(1000, lambda: (root.destroy()))

def show_done():
    alert = tk.Toplevel()
    alert.configure(bg="green")
    alert.relief = tk.RAISED
    alert.title("Coś tam idzie")
    alert.geometry("100x100")

    label = tk.Label(alert, text=f"👍", bg='green',font=("Comic Sans MS", 50))

    label.pack(pady=20)

    alert.after(1000, lambda: (root.destroy()))

# ---------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
time_start = time.time()
FILENAME = f'../NEW_saved_results/NEW_data_demand_recurrent_SmoothL1Loss_AdamW_64_500_const_epoch_small.csv'


def generate_vectors(vectors_number, vector_size):
    return torch.rand((vectors_number, vector_size), dtype=torch.float32, device=device)


def calculate_distance(x_dataset, metric='euclidean'):
    if metric == 'euclidean':
        return torch.sqrt(torch.sum(torch.pow(x_dataset[:, :, 0] - x_dataset[:, :, 1], 2), dim=1)).to(device)
    elif metric == 'manhattan':
        return torch.sqrt(torch.sum(x_dataset[:, :, 0] - x_dataset[:, :, 1], dim=1)).to(device)
    elif metric == 'cosine':
        return torch.sum(torch.mm(x_dataset[:, :, 0], x_dataset[:, :, 0]), dim=1).to(device)


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim_r, hidden_dim_fc=64, num_layers_recurrent=1, num_layers_fc=2):
        super(LSTMModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim_r = hidden_dim_r
        self.hidden_dim_fc = hidden_dim_fc
        self.num_layers_recurrent = num_layers_recurrent
        self.num_layers_fc = num_layers_fc

        self.rnn = nn.LSTM(2, hidden_dim_r, num_layers_recurrent, batch_first=True)

        module_list = []
        if self.num_layers_recurrent == 1:
            module_list.append(nn.Linear(self.hidden_dim_r, 1))
        else:
            for i in range(self.num_layers_fc):
                if i == 0:
                    module_list.append(nn.Linear(hidden_dim_r, hidden_dim_fc))
                elif i != self.num_layers_fc - 1:
                    module_list.append(nn.Linear(hidden_dim_fc, hidden_dim_fc))
                else:
                    module_list.append(nn.Linear(hidden_dim_fc, 1))
        self.fc = nn.ModuleList(module_list)

    def forward(self, input):
        out, _ = self.rnn(input)
        out = out[:, -1, :]
        for i in range(self.num_layers_fc - 1):
            # was leaky_relu, now tanh
            out = nn.functional.tanh(self.fc[i](out))
        return self.fc[-1](out)




def create_dataloader(dataset, pairs,metric):
    x_pairs = torch.stack([torch.stack((dataset[i], dataset[j]), dim=0) for i, j in pairs], dim=0)
    distances = calculate_distance(x_pairs, metric=metric).unsqueeze(-1)
    tensor_dataset = TensorDataset(x_pairs, distances)
    return DataLoader(tensor_dataset, batch_size=64, shuffle=True)



def train(model, input_dim, optimizer, criterion, max_epochs, data_demand, factor, metric, loss_tolerance,
          batch_size=64, mode='rnn'):
    max_distance = math.sqrt(input_dim)
    print(f'\033[38;2;255;255;255m')
    print("Training model for parameters:")
    print("input_dim:", input_dim)
    print("hidden_dim_r:", model.hidden_dim_r)
    print("hidden_dim_fc:", model.hidden_dim_fc)
    print("num_layers_recurrent:", model.num_layers_recurrent)
    print("num_layers_fc:", model.num_layers_fc)
    print("data_demand:", data_demand)
    print("max_distance:", max_distance)
    print("factor", factor)
    print("loss_tolerance:", loss_tolerance)
    print("batch_size:", batch_size)

    best_loss = float("inf")
    best_loss_epoch = 0
    dataset = generate_vectors(data_demand, input_dim)


    pairs = torch.tensor([(i, j) for i in range(len(dataset)) for j in range(i + 1, len(dataset))])
    dataloader = create_dataloader(dataset, pairs,metric)
    model.train()
    for epoch in range(max_epochs):
        epoch_loss = 0
        dataloaderCounter = 0
        sampleCounter=1
        print(f'\033[38;2;255;50;255m')
        topFive = [float("inf"),float("inf"),float("inf"),float("inf"),float("inf")]
        for x_data, y_data in dataloader:
            if mode == 'rnn':
                x_data = x_data.permute(0, 2, 1)

            running_avg_norm = 0.0
            alpha = 0.9  # Współczynnik uśredniania

            optimizer.zero_grad()
            output = model(x_data)
            loss = criterion(output, y_data)
            loss.backward()
            # Obliczanie normy gradientu
            total_norm = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            running_avg_norm = alpha * running_avg_norm + (1 - alpha) * total_norm

            # Gradient clipping z adaptacyjnym max_norm
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=running_avg_norm * 1.5)
            optimizer.step()
            if loss.item() < topFive[4]:
                topFive[4] = loss.item()
                topFive.sort()
            XText = f'\rTrain iteration: {dataloaderCounter}/{len(dataloader)},  Loss: {loss.item():.8f}'
            if dataloaderCounter%10==0:
                print(XText,end='')
            dataloaderCounter += 1
        for value in topFive:
            epoch_loss +=value/5
        # Aktualizacja najlepszego wyniku
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_loss_epoch = epoch
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
                'epoch': best_loss_epoch,
            }, "best_model.pth")

        # other funny color effects
        # myHSV = f'0.5;1;1'
        # myHSV2 = f'{hsvConverterV(epoch/500,0.1,1)[0]};{hsvConverterV(epoch/100,1,0.5)[1]};{hsvConverterV(epoch/100,1,0.5)[2]}'

        rgbText = f'\rEpoch: {epoch}/{max_epochs},  Loss: {epoch_loss:.8f}, Best loss: {best_loss}, Best loss epoch: {best_loss_epoch}'
        counter = 0
        for letter in rgbText:
            print(
                f'\033[38;2;{hsvConverterV(epoch / 50 + counter, 1, 1)[0]};{hsvConverterV(epoch / 50 + counter, 1, 1)[1]};{hsvConverterV(epoch / 50 + counter, 1, 1)[2]}m{letter}',
                end='')
            counter += 0.005
        print(f'\033[38;2;255;255;255m Progress:\t {min(100, 100 * (loss_tolerance / best_loss)):.2f} %', end='\t')
        goalReached = (int(50 * (loss_tolerance / best_loss))%50) * '■'
        bar = ((50 - int(50 * (loss_tolerance / best_loss)))%50) * '■'

        for letter in goalReached:
            print(f'\033[38;2;0;255;0m{letter}', end='')
        for letter in bar:
            print(f'\033[38;2;0;0;0m{letter}', end='')

        # print(f'\rEpoch: {epoch}/{max_epochs}, Loss: {loss.item():.8f}, Best loss: {best_loss}, Best loss as distance(best_loss * max_distance): {best_loss * max_distance}, Best loss epoch: {best_loss_epoch}', end='')
        if epoch - best_loss_epoch >= 10:
            # show_alert(f'input dim:{input_dim}\nfactor:{factor} ')
            print('\nNo progress, stopping training...')
            # root.mainloop()
            return model, max_epochs, best_loss
        # for constatnt epoch number
        # if best_loss < loss_tolerance:
        #     print('\nStopping training...')
        #     # show_done()
        #     # root.mainloop()
        #
        #     return model, epoch, best_loss
        # show_done()
        # root.mainloop()
    return model, max_epochs, best_loss


def test(model, input_dim, criterion, test_dataset_size, metric, mode='rnn'):
    print("\nTesting model...")
    # Wczytanie modelu i jego stanu
    checkpoint = torch.load("best_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    with torch.no_grad():
        # Generowanie datasetu testowego
        dataset = generate_vectors(test_dataset_size, input_dim)
        pairs = torch.tensor([(i, j) for i in range(len(dataset)) for j in range(i + 1, len(dataset))])
        dataloader = create_dataloader(dataset, pairs,metric)
        total_loss = 0
        dataloaderCounter=0
        print(f'\033[38;2;255;50;255m')

        for x_data, y_data in dataloader:
            if mode == 'rnn':
                x_data = x_data.permute(0, 2, 1)
            output = model(x_data)
            loss = criterion(output, y_data)
            total_loss += loss.item()
            XText = f'\rTest iteration: {dataloaderCounter}/{len(dataloader)},  Loss: {loss.item():.8f}'
            if dataloaderCounter % 10 == 0:
                print(XText, end='')
            dataloaderCounter += 1
        avg_loss = total_loss / dataloaderCounter
        print(f"Test loss: {avg_loss:.8f}")
        return avg_loss


def hsvConverterV(h,s,v):
    rgb = colorsys.hsv_to_rgb(h,s,v)
    return tuple(int(c*255)for c in rgb)



def search_data_demand_recurrent(dims):
    print("Using device:", device)
    tests_number = 5
    extrapolated_data_demand = [100]
    test_dataset_size = 100
    hidden_dim_r = 64 # was 64
    hidden_dim_fc = 500 # was 500
    num_layers_recurrent = 3
    num_layers_fc = 3
    metric = 'euclidean'

    input_dims = [dims]
    lr = 0.001
    criterion = nn.SmoothL1Loss().to(device)
    ### try MSELoss, SmoothL1Loss

    # dataset.length = extrapolated_data_demand * factor
    min_factor = 1.0
    max_factor = 19.0
    step = 2.0
    max_epochs = 5

    loops = math.ceil((max_factor - min_factor) / step)

    for _ in range(tests_number):
        for input_dim in input_dims:
            loss_tolerance = 0.01

            for i in range(loops + 1):
                start_time_loop = time.time()
                model = LSTMModel(input_dim, hidden_dim_r, hidden_dim_fc, num_layers_recurrent, num_layers_fc).to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
                ### try RMSprop, AdamW,

                data_demand = extrapolated_data_demand[input_dims.index(input_dim)]
                factor = min_factor + i * step
                data_demand = math.floor(data_demand * factor)
                model, epoch, train_loss = train(model, input_dim, optimizer, criterion, max_epochs, data_demand, factor, metric, loss_tolerance=loss_tolerance, batch_size=64)
                test_loss = test(model, input_dim, criterion, test_dataset_size, metric)
                max_distance = math.sqrt(input_dim)
                train_loss_distance = train_loss*max_distance
                test_loss_distance = test_loss*max_distance
                file_exists = os.path.isfile(FILENAME)
                with open(FILENAME, 'a') as file:
                    if not file_exists:
                        file.write("Input_dim,Hidden_dim_r,Hidden_dim_fc,Num_layers_recurrent,Num_layers_fc,Loss_tolerance,Data_demand,Dd_factor,Train_loss,Test_loss,Train_loss_distance,Test_loss_distance,Epochs,Time\n")
                    file.write(f"{model.input_dim},{model.hidden_dim_r},{model.hidden_dim_fc},{model.num_layers_recurrent},{model.num_layers_fc},{loss_tolerance},{data_demand},{factor},{train_loss},{test_loss},{train_loss_distance},{test_loss_distance},{epoch+1},{int(time.time()-start_time_loop)}\n")


if __name__ == '__main__':
    for loopdims in range(14,30,3):
        search_data_demand_recurrent(loopdims)
