import torch
import torch.nn as nn
from torch.nn import functional as F

hidden_dim = 128


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


# Actor-Critic
class PolicyNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        nn.init.normal_(self.fc1.weight,0, 0.1)
        nn.init.normal_(self.fc2.weight,0, 0.1)


    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = F.softmax(x, dim=-1)+1e-9

        return x


class ValueNet(nn.Module):
    def __init__(self, input_dim):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
