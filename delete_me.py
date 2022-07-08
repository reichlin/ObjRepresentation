import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt


class Block(nn.Module):

    def __init__(self, in_dim, out_dims):
        super().__init__()

        self.fc1 = nn.Linear(in_dim, 64)

        self.module1 = nn.Sequential(nn.Linear(64, 64),
                                     nn.ReLU(),
                                     nn.Linear(64, 64),
                                     nn.ReLU())

        self.module2 = nn.Sequential(nn.Linear(64, 64),
                                     nn.ReLU(),
                                     nn.Linear(64, out_dims))

    def forward(self, x):
        z = F.relu(self.fc1(x))
        z = self.module1(z) + z
        return self.module2(z)


class f_network(nn.Module):

    def __init__(self, a_dim, n_layers):
        super().__init__()

        self.blk1 = Block(a_dim, 64)
        self.blk2 = Block(64, 64)
        self.blk3 = Block(64, a_dim)

    def forward(self, h):
        z = self.blk1(h)
        z = self.blk2(z) + z
        return self.blk3(z)


data = None
for i in range(28):
    for j in range(28):
        data = np.array([[i, j]]) if data is None else np.concatenate((data, np.array([[i, j]])), 0)


rnd_idx = np.arange(28*28)
np.random.shuffle(rnd_idx)
y_real = torch.from_numpy(data).float()
x = torch.from_numpy(data[rnd_idx]).float()* 0.1


writer = SummaryWriter("./logs_balanced/delete_me4")


f = f_network(2, 8)
optimizer = optim.Adam(f.parameters(), lr=1e-3)


for e in tqdm(range(10000000)):

    y_hat = f(x) * 28
    y = y_real + torch.from_numpy(np.random.randint(-2, 2, size=(28*28, 2))).float()
    L = torch.mean(torch.sum((y_hat - y)**2, -1))

    optimizer.zero_grad()
    L.backward()
    optimizer.step()

    writer.add_scalar("loss", L.detach().cpu().numpy(), e)

    fig = plt.figure()
    plt.scatter(y_hat[:, 0].detach().cpu().numpy(), y_hat[:, 1].detach().cpu().numpy())
    plt.scatter(y[:, 0].detach().cpu().numpy(), y[:, 1].detach().cpu().numpy(), alpha=0.5)
    writer.add_figure("y_dist", fig, e)


















