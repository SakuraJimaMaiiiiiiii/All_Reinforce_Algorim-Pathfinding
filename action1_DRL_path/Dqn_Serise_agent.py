import random
import torch
from torch.nn import functional as F
from model import DuelingDQN, DNN
from replay_memory import ReplayMemory, ReplayMemory_Per


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
EPSILON_END = 0.01
EPSILON_START = 1.0
EPSILON_DECAY = 0.95
learn_rate = 0.003
gamma = 0.9  # 折扣因子

state_dim = 3
action_dim = 26


class DuelingQN:
    def __init__(self, env):
        self.env = env
        self.epsilon = 1.0
        self.memory = ReplayMemory(10000)
        self.steps_done = 0
        self.name = 'DuelingQN'
        self.policy_net = DuelingDQN(state_dim, action_dim).to(device)
        self.target_net = DuelingDQN(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learn_rate)

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
            action = random.randint(0, action_dim - 1)
        return action

    def learn(self):
        if len(self.memory) < batch_size:
            return
        transitions = self.memory.sample(batch_size)
        batch = list(zip(*transitions))
        state_batch = torch.tensor(batch[0], dtype=torch.float32).to(device)    # [64,3]
        action_batch = torch.tensor(batch[1], dtype=torch.long).view(-1, 1).to(device)
        reward_batch = torch.tensor(batch[2], dtype=torch.float32).to(device)
        next_state_batch = torch.tensor(batch[3], dtype=torch.float32).to(device)
        done_batch = torch.tensor(batch[4], dtype=torch.float32).to(device)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)    # [64,1]


        with torch.no_grad():
            next_state_values = self.target_net(next_state_batch).max(1)[0].detach()   # [64]
            expected_state_action_values = reward_batch + (1.0 - done_batch) * gamma * next_state_values  #[64]

        loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_model()
        self.steps_done += 1


    def update_model(self):
        if self.steps_done % 10 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())


'''
###################################################################################################
'''


class DuelingQN_Prememory:
    def __init__(self, env):
        self.env = env
        self.epsilon = 1.0
        self.memory = ReplayMemory_Per(10000)
        self.steps_done = 0
        self.name = 'PerDuelingQN'
        self.policy_net = DuelingDQN(state_dim, action_dim).to(device)
        self.target_net = DuelingDQN(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learn_rate)

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

        loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_model()
        self.steps_done += 1

    def update_model(self):
        if self.steps_done % 10 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())


'''
#######################################################################################################
'''


class DoubleDQN:
    def __init__(self, env):
        self.env = env
        self.epsilon = 1.0
        self.memory = ReplayMemory(10000)
        self.memory_counter = 0
        self.steps_done = 0
        self.name = 'DoubleDQN'

        self.policy_net = DNN(state_dim, action_dim).to(device)
        self.target_net = DNN(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learn_rate)

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
            action = random.randint(0, action_dim - 1)
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

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)  # [64,1]

        with torch.no_grad():
            next_action_batch = torch.unsqueeze(self.policy_net(next_state_batch).max(1)[1], 1)
            next_state_values = self.target_net(next_state_batch).gather(1, next_action_batch)   # [64,1]
            expected_state_action_values = reward_batch.unsqueeze(1) + ((1.0 - done_batch).unsqueeze(1)) * next_state_values * gamma


        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()



        self.update_model()
        self.steps_done += 1

    def update_model(self):
        if self.steps_done % 10 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())


'''
########################################################################################################
'''


class DQN:
    def __init__(self, env):
        self.env = env
        self.epsilon = 1.0
        self.memory = ReplayMemory(10000)
        self.steps_done = 0
        self.name = 'DQN'

        self.policy_net = DNN(state_dim, action_dim).to(device)
        self.target_net = DNN(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learn_rate)

    def choose_action(self, state):
        sample = random.random()
        self.epsilon = max(0.01, self.epsilon * 0.995)
        if sample > self.epsilon:
            with torch.no_grad():
                q_values = self.policy_net(torch.tensor(state, dtype=torch.float32).to(device))
                action = torch.argmax(q_values).item()  # int

        else:
            action = random.randint(0, action_dim - 1)
        return action

    def store_transition(self, state, action, reward, next_stata, done):
        self.memory.push((state, action, reward, next_stata, done))

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
            expected_state_action_values = reward_batch + (1.0 - done_batch) * gamma * next_state_values

        loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


        self.update_model()
        self.steps_done += 1

    def update_model(self):
        if self.steps_done % 10 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())


'''
###########################################################################################################
'''


