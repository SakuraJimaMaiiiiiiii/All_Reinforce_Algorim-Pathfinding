import random
import torch
from torch.nn import functional as F
from model import DuelingDQN, DNN, PolicyNet, ValueNet
from replay_memory import ReplayMemory, ReplayMemory_Per
import numpy as np

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
        self.steps_done += 1

    def update_target_model(self):
        if self.steps_done % 10 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())


''''''''


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

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # next_action_batch = torch.unsqueeze(self.model_target(next_state_batch).max(1)[1], 1)
        next_action_batch = torch.unsqueeze(self.policy_net(next_state_batch).max(1)[1], 1)
        next_state_values = self.target_net(next_state_batch).gather(1, next_action_batch)
        expected_state_action_values = reward_batch + (1.0 - done_batch) * next_state_values * gamma

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def update_target_model(self):
        if self.steps_done % 10 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())


''''''


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
                action = torch.argmax(q_values).item()
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
        self.steps_done += 1

    def update_target_model(self):
        if self.steps_done % 10 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())


''''''

class Actor_2_Critic:
    def __init__(self, env):
        self.env = env
        self.epsilon = 1.0
        self.memory = ReplayMemory_Per(10000)
        self.steps_done = 0
        self.name = 'Actor_2_Critic'
        self.actor_lr = 0.001
        self.critic_lr = 0.001


        self.actor_net = PolicyNet(state_dim, action_dim).to(device)  # 策略网络
        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=self.actor_lr)

        self.critic_net = ValueNet(state_dim).to(device)  # 价值网络
        self.critic_optimizer = torch.optim.Adam(self.critic_net.parameters(), lr=self.critic_lr)

        # self.critic_delay_net = ValueNet(state_dim).to(device)
        # self.critic_delay_net.load_state_dict(self.critic_net.state_dict())


    def store_transition(self, state, action, reward, next_stata, done):
        self.memory.push((state, action, reward, next_stata, done))


    def choose_action(self, state):
        # 动作价值函数 当下状态下的动作分布
        with torch.no_grad():
            prob = self.actor_net(torch.tensor(state, dtype=torch.float32).to(device))
            if torch.isnan(prob).any():
                print(prob)
                print("NaN detected in prob!")

            # 创建以prob为标准类型的数据分布
            action_list = torch.distributions.Categorical(prob)

            # 根据概率随机采样
            action = action_list.sample().item()

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



        # 预测的当前时刻的state_value
        td_value = self.critic_net(state_batch)
        #目标的当前时刻的state_value
        td_target = reward_batch.unsqueeze(1) + gamma * self.critic_net(next_state_batch) * ((1 - done_batch).unsqueeze(1))
        # td_target = reward_batch.unsqueeze(1) + gamma * self.critic_delay_net(next_state_batch) * (
        #     (1 - done_batch).unsqueeze(1))
        #时序差分的误差计算 目标的state_value与预测的state_value之差
        td_error = td_target - td_value                 #  tderror = r + γQ(s') - Q(s)
        #对每个状态对应得动作价值用log函数
        log_prob = torch.log(self.actor_net(state_batch).gather(1, action_batch))
        #策略梯度损失
        actor_loss = torch.mean(-log_prob * td_error.detach())
        #值函数损失，预测值与目标值之间
        critic_loss = torch.mean(F.mse_loss(self.critic_net(state_batch), td_target.detach()))

        self.actor_optimizer.zero_grad()  # 策略梯度网络的优化器
        self.critic_optimizer.zero_grad()  # 价值网络的优化器
        actor_loss.backward()
        critic_loss.backward()

        self.actor_optimizer.step()
        self.critic_optimizer.step()


    def update_target_model(self):
        # for param, param_delay in zip(self.critic_net.parameters(), self.critic_delay_net.parameters()):
        #     value = param_delay.data *0.7 + param*0.3
        #     param_delay.data.copy_(value)
        pass



#############################


class Actor_Critic:
    def __init__(self, env):
        self.env = env
        self.epsilon = 1.0
        self.memory = ReplayMemory_Per(10000)
        self.steps_done = 0
        self.name = 'Actor_Critic'
        self.actor_lr = 0.0001
        self.critic_lr = 0.0001

        self.actor_net = PolicyNet(state_dim, action_dim).to(device)  # 策略网络
        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=self.actor_lr)

        self.critic_net = ValueNet(state_dim).to(device)  # 价值网络
        self.critic_optimizer = torch.optim.Adam(self.critic_net.parameters(), lr=self.critic_lr)

        # self.critic_delay_net = ValueNet(state_dim).to(device)
        # self.critic_delay_net.load_state_dict(self.critic_net.state_dict())


    def store_transition(self, state, action, reward, next_stata, done):
        self.memory.push((state, action, reward, next_stata, done))


    def choose_action(self, state):
        # 动作价值函数 当下状态下的动作分布
        with torch.no_grad():
            prob = self.actor_net(torch.tensor(state, dtype=torch.float32).to(device))
            if torch.isnan(prob).any():
                print(prob)
                print("NaN detected in prob!")

            # 创建以prob为标准类型的数据分布
            action_list = torch.distributions.Categorical(prob)

            # 根据概率随机采样
            action = action_list.sample().item()

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



        # 预测的当前时刻的state_value
        td_value = self.critic_net(state_batch)


        #目标的当前时刻的state_value
        td_target = reward_batch.unsqueeze(1) + gamma * self.critic_net(next_state_batch) * ((1 - done_batch).unsqueeze(1))

        # td_target = reward_batch.unsqueeze(1) + gamma * self.critic_delay_net(next_state_batch) * ((1 - done_batch).unsqueeze(1))


        #时序差分的误差计算 目标的state_value与预测的state_value之差
        td_error = td_target - td_value



        #对每个状态对应得动作价值用log函数
        log_prob = torch.log(self.actor_net(state_batch).gather(1, action_batch))


        #策略梯度损失
        actor_loss = torch.mean(-log_prob * td_value.detach())



        #值函数损失，预测值与目标值之间
        critic_loss = torch.mean(F.mse_loss(self.critic_net(state_batch), td_target.detach()))


        self.actor_optimizer.zero_grad()  # 策略梯度网络的优化器
        self.critic_optimizer.zero_grad()  # 价值网络的优化器
        actor_loss.backward()
        critic_loss.backward()

        self.actor_optimizer.step()
        self.critic_optimizer.step()


    def update_target_model(self):
        # for param, param_delay in zip(self.critic_net.parameters(), self.critic_delay_net.parameters()):
        #     value = param_delay.data *0.7 + param*0.3
        #     param_delay.data.copy_(value)
        pass


###################
#
# class DDPG:


##################
# class PPO:


##################
