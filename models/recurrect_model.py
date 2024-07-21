import torch.nn as nn


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
        for layer in self.fc:
            out = layer(out)
        return out


class ExperimentalRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers_recurrent=1, num_layers_fc=2):
        super(ExperimentalRNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers_recurrent = num_layers_recurrent
        self.num_layers_fc = num_layers_fc

        self.rnn = nn.LSTM(2, hidden_dim, num_layers_recurrent, batch_first=True)

        inner_dim = 512
        module_list = []
        for i in range(self.num_layers_fc - 1):
            if i == 0:
                module_list.append(nn.Linear(hidden_dim, inner_dim))
            else:
                module_list.append(nn.Linear(inner_dim, inner_dim))
        module_list.append(nn.Linear(inner_dim, 1))
        self.fc = nn.ModuleList(module_list)

    def forward(self, input):
        out, _ = self.rnn(input)
        out = out[:, -1, :]
        for layer in self.fc:
            out = layer(out)
        return out
