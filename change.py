import random
import torch
import torch.optim as optim
from model import DuelingDQN
from replay_memory import ReplayMemory,ReplayMemory_Per
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32
EPSILON_END = 0.01
EPSILON_START = 1.0
EPSILON_DECAY = 0.95
learn_rate = 0.001
gamma = 0.9  #折扣因子

class DuelingQN:
    def __init__(self, env):
        self.env = env
        self.epsilon = 1.0
        self.policy_net = DuelingDQN(3, 26).to(device)
        self.target_net = DuelingDQN(3, 26).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learn_rate)
        self.memory = ReplayMemory(10000)
        self.steps_done = 0

    def store_transition(self, state, action, reward, next_stata,done):
        self.memory.push((state, action, reward, next_stata, done))

    def choose_action(self, state):
        sample = random.random()
        self.epsilon = max(0.01, self.epsilon * 0.995)
        if sample > self.epsilon:
            with torch.no_grad():
                q_values = self.policy_net(torch.tensor(state, dtype=torch.float32).to(device))
                action = torch.argmax(q_values).item()
        else:
            action = random.randint(0, 25)
        return action


    def learn(self):
        if len(self.memory) < batch_size:
            return
        transitions = self.memory.sample(batch_size)
        batch = list(zip(*transitions))
        state_batch = torch.tensor(batch[0], dtype=torch.float32).to(device)
        action_batch = torch.tensor(batch[1], dtype=torch.long).view(-1, 1).to(device)
        reward_batch = torch.tensor(batch[2], dtype=torch.float32).to(device)
        next_state_batch = torch.tensor(batch[3], dtype=torch.float32).to(device)
        done_batch = torch.tensor(batch[4], dtype=torch.float32).to(device)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        with torch.no_grad():
            next_state_values = self.target_net(next_state_batch).max(1)[0].detach()
            expected_state_action_values = reward_batch + (1.0 - done_batch) * 0.99 * next_state_values

        loss = torch.nn.functional.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.steps_done += 1

    def update_target_model(self):
        if self.steps_done % 10 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())



''''''''
class PerDuelingQN:
    def __init__(self, env):
        self.env = env
        self.epsilon = 1.0
        self.memory = ReplayMemory_Per(10000)
        self.memory_counter = 0

        self.policy_net = DuelingDQN(3, 26).to(device)
        self.target_net = DuelingDQN(3, 26).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learn_rate)
        self.steps_done = 0

    def store_transition(self, state, action, reward, next_stata,done):
        self.memory.push((state, action, reward, next_stata, done))


    def choose_action(self, state):
        sample = random.random()
        self.epsilon = max(0.01, self.epsilon * 0.995)
        if sample > self.epsilon:
            with torch.no_grad():
                q_values = self.policy_net(torch.tensor(state, dtype=torch.float32).to(device))
                action = torch.argmax(q_values).item()
        else:
            action = random.randint(0, 25)
        return action


    def learn(self):
        if self.memory.size() < batch_size:
            return
        idxs, transitions = self.memory.sample(batch_size)

        batch = list(zip(*transitions))
        state_batch = torch.tensor(batch[0], dtype=torch.float32).to(device)
        action_batch = torch.tensor(batch[1], dtype=torch.long).view(-1, 1).to(device)
        reward_batch = torch.tensor(batch[2], dtype=torch.float32).to(device)
        next_state_batch = torch.tensor(batch[3], dtype=torch.float32).to(device)
        done_batch = torch.tensor(batch[4], dtype=torch.float32).to(device)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        with torch.no_grad():
            next_state_values = self.target_net(next_state_batch).max(1)[0].detach()
            expected_state_action_values = reward_batch + (1.0 - done_batch) * gamma * next_state_values

        td_errors = (state_action_values - expected_state_action_values.unsqueeze(1)).detach().squeeze().tolist()
        self.memory.update(idxs, td_errors)  # update td error

        loss = torch.nn.functional.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.steps_done += 1

    def update_target_model(self):
        if self.steps_done % 10 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())



class DQN:
    def __init__(self, env):
        self.env = env
        self.epsilon = 1.0
        self.memory = ReplayMemory(10000)
        self.memory_counter = 0
        self.steps_done = 0


        self.policy_net = DNN(3, 26).to(device)
        self.target_net = DNN(3, 26).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learn_rate)


    def store_transition(self, state, action, reward, next_stata, done):
        self.memory.push((state, action, reward, next_stata, done))



    def choose_action(self, state):
        sample = random.random()
        self.epsilon = max(0.01, self.epsilon * 0.995)
        if sample > self.epsilon:
            with torch.no_grad():
                q_values = self.policy_net(torch.tensor(state, dtype=torch.float32).to(device))
                action = torch.argmax(q_values).item()
        else:
            action = random.randint(0, 25)
        return action


    def learn(self):
        if len(self.memory) < batch_size:
            return
        transitions = self.memory.sample(batch_size)
        batch = list(zip(*transitions))
        state_batch = torch.tensor(batch[0], dtype=torch.float32).to(device)
        action_batch = torch.tensor(batch[1], dtype=torch.long).view(-1, 1).to(device)
        reward_batch = torch.tensor(batch[2], dtype=torch.float32).to(device)
        next_state_batch = torch.tensor(batch[3], dtype=torch.float32).to(device)
        done_batch = torch.tensor(batch[4], dtype=torch.float32).to(device)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        with torch.no_grad():
            next_state_values = self.target_net(next_state_batch).max(1)[0].detach()
            expected_state_action_values = reward_batch + (1.0 - done_batch) * 0.99 * next_state_values

        loss = torch.nn.functional.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.steps_done += 1

    def update_target_model(self):
        if self.steps_done % 10 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
