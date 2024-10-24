import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

hidden_dim = 256

'''
辅助函数 
'''


# 防止梯度爆炸
def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DuelingDQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_value = nn.Linear(hidden_dim, 1)
        self.fc_advantage = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        value = self.fc_value(x)
        advantage = self.fc_advantage(x)
        return value + (advantage - advantage.mean())


class DNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_value = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        value = self.fc_value(x)

        return value


# AC actor net
class ActorNet1(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorNet1, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.ln2 = nn.LayerNorm(output_dim)
        nn.init.normal_(self.fc1.weight, 0, 0.1)
        nn.init.normal_(self.fc2.weight, 0, 0.1)

    def forward(self, x):
        x = torch.relu(self.ln1(self.fc1(x)))
        x = torch.relu(self.ln2(self.fc2(x)))
        x = F.softmax(x, dim=-1) + 1e-9  # len : output dim
        return x



# AC critic net
class CriticNet(nn.Module):
    def __init__(self, input_dim):
        super(CriticNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        nn.init.normal_(self.fc1.weight, 0, 0.1)
        nn.init.normal_(self.fc2.weight, 0, 0.1)

    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.fc2(x)
        return x


# SAC critic net
class CriticNet2(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CriticNet2, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        nn.init.normal_(self.fc1.weight, 0, 0.1)
        nn.init.normal_(self.fc2.weight, 0, 0.1)

    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.fc2(x)
        return x




