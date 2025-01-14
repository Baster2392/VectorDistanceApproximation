import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

"""
After all TODO's fine-tune the two problematic modules, discover optimal hyperparameters
Perfectly also analyse following relations:
1. dimensionality vs complexity
2. dimensionality vs dataset size
Take into considerations both the amount of neurons in each layer and the number of layers.
"""


########################################
# Neural Modules
########################################


class NeuralSummation(nn.Module):
    """
    Neural approximation of z -> sum(z)
    This should be perfect, always.
    Does not need to be trained.
    """

    def __init__(self, n):
        super().__init__()
        self.linear = nn.Linear(n, 1, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.ones(1, n))

    def forward(self, z):
        return self.linear(z)


class NeuralSubtraction(nn.Module):
    """
    Neural approximation of subtraction.
    x, y -> x - y
    This, simillarlly to NeuralSummation can be performed perfectly and without training
    i.e we know the perfect weights and biases for the neurons in the module.
    """

    def __init__(self, n):
        super().__init__()
        self.linear = nn.Linear(2 * n, n, bias=False)
        with torch.no_grad():
            W = torch.zeros(n, 2 * n)
            W[:, :n] = torch.eye(n)
            W[:, n:] = -torch.eye(n)
            self.linear.weight.copy_(W)

    def forward(self, x, y):
        concat = torch.cat([x, y], dim=-1)
        return self.linear(concat)


class NeuralSquare(nn.Module):
    """
    Neural approximation of z -> z^2
    This is
    """

    def __init__(self, n, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n)
        )

    def forward(self, z):
        return self.net(z)


class NeuralSqrt(nn.Module):
    """
    Simple neural approximation of of s -> sqrt(s)
    """
    
    def __init__(self, n=1, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n)
        )

    def forward(self, s):
        return self.net(s)


########################################
# Datasets
########################################

class SimpleDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


# could modify the generation functions to use different distributions than randn(normal distrubiton)
def create_subtraction_dataset(n_samples=1000, n=3):
    x = torch.randn(n_samples, n)
    y = torch.randn(n_samples, n)
    targets = x - y
    inputs = torch.cat([x, y], dim=-1)
    return SimpleDataset(inputs, targets)


def create_square_dataset(n_samples=1000, n=3):
    z = torch.randn(n_samples, n)
    z_sq = z ** 2
    return SimpleDataset(z, z_sq)


def create_summation_dataset(n_samples=1000, n=3):
    z = torch.randn(n_samples, n)
    s = torch.sum(z, dim=-1, keepdim=True)
    return SimpleDataset(z, s)


def create_sqrt_dataset(n_samples=1000):
    s = torch.rand(n_samples, 1) * 100.0
    s_sqrt = torch.sqrt(s)
    return SimpleDataset(s, s_sqrt)


########################################
# Training
########################################

def train_single_input_module(module, loader, epochs=10, lr=0.01):
    """
    Train a module with a single input (z -> z^2, z -> sum(z), s -> sqrt(s))
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(module.parameters(), lr=lr)
    module.train()
    for epoch in range(epochs):
        total_loss = 0
        for inp, tgt in loader:
            optimizer.zero_grad()
            out = module(inp)
            loss = criterion(out, tgt)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inp.size(0)
        avg_loss = total_loss / len(loader.dataset)
        # print(f"[Epoch {epoch+1}/{epochs}] Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    torch.manual_seed(0)
    n = 3  # dimension of vectors

    # 1. Initialize Subtraction Module (will be perfect if correctly initialized)
    sub_dataset = create_subtraction_dataset(n_samples=2000, n=n)
    sub_loader = DataLoader(sub_dataset, batch_size=64, shuffle=True)
    neural_sub = NeuralSubtraction(n)

    # 2. Train Square Module
    square_dataset = create_square_dataset(n_samples=2000, n=n)
    square_loader = DataLoader(square_dataset, batch_size=64, shuffle=True)
    neural_square = NeuralSquare(n)
    train_single_input_module(neural_square, square_loader, epochs=20)

    # 3. Train Summation Module (no need for training)
    sum_dataset = create_summation_dataset(n_samples=2000, n=n)
    sum_loader = DataLoader(sum_dataset, batch_size=64, shuffle=True)
    neural_sum = NeuralSummation(n)

    # 4. Train Sqrt Module
    sqrt_dataset = create_sqrt_dataset(n_samples=2000)
    sqrt_loader = DataLoader(sqrt_dataset, batch_size=64, shuffle=True)
    neural_sqrt = NeuralSqrt()
    train_single_input_module(neural_sqrt, sqrt_loader, epochs=20)

    x_test = torch.tensor([[1.0, 2.0, 3.0]])
    y_test = torch.tensor([[4.0, 2.0, 0.0]])
    diff_pred = neural_sub(x_test, y_test)

    z_sq_pred = neural_square(diff_pred)
    z_sum_pred = neural_sum(z_sq_pred)
    s_sqrt_pred = neural_sqrt(z_sum_pred)
    print(f'x: {x_test} - y: {y_test}  == z: {diff_pred}    [should be  {x_test - y_test}]')
    print(f'z^2: {z_sq_pred}      [should be  {diff_pred ** 2}]')
    sum = torch.sum(z_sq_pred, dim=-1, keepdim=True)
    print(f'sum(z): {z_sum_pred}      [should be  {sum}]')
    print(f'sqrt(z): {s_sqrt_pred}      [should be  {torch.sqrt(z_sum_pred)}]')

