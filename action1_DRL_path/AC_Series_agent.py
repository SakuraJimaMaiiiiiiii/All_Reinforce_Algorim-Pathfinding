import torch
from torch.nn import functional as F
from model import  ActorNet1, CriticNet, CriticNet2
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



class Actor_Critic_Memory:
    def __init__(self, env):
        self.env = env
        self.epsilon = 1.0
        self.memory = ReplayMemory(10000)
        self.steps_done = 0
        self.name = 'Actor_Critic_Memory'
        self.actor_lr = 0.0001
        self.critic_lr = 0.0001

        self.actor_net = ActorNet1(state_dim, action_dim).to(device)  # 策略网络
        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=self.actor_lr)

        self.critic_net = CriticNet(state_dim).to(device)  # 价值网络
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
        action = action_list.sample().item()  # int(0-25),int

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

        # 预测的当前时刻的state_value
        td_value = self.critic_net(state_batch)

        # 目标的当前时刻的state_value
        td_target = reward_batch.unsqueeze(1) + gamma * self.critic_net(next_state_batch) * (
            (1 - done_batch).unsqueeze(1))

        # td_target = reward_batch.unsqueeze(1) + gamma * self.critic_delay_net(next_state_batch) * ((1 - done_batch).unsqueeze(1))

        # 对每个状态对应得动作价值用log函数
        log_prob = torch.log(self.actor_net(state_batch).gather(1, action_batch))

        # 策略梯度损失
        actor_loss = -torch.mean(log_prob * td_value.detach())

        # 值函数损失，预测值与目标值之间
        critic_loss = torch.mean(F.mse_loss(self.critic_net(state_batch), td_target.detach()))

        self.actor_optimizer.zero_grad()  # 策略梯度网络的优化器
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()  # 价值网络的优化器
        critic_loss.backward()
        self.critic_optimizer.step()

        self.update_model()

        self.steps_done += 1

    def update_model(self):
        if self.steps_done % 10 == 0:
            # for param, param_delay in zip(self.critic_net.parameters(), self.critic_delay_net.parameters()):
            #     value = param_delay.data *0.7 + param*0.3
            #     param_delay.data.copy_(value)
            pass



class Actor_Critic_PreMemory:
    def __init__(self, env):
        self.env = env
        self.epsilon = 1.0
        self.memory = ReplayMemory_Per(10000)
        self.steps_done = 0
        self.name = 'Actor_Critic_PreMemory'
        self.actor_lr = 0.0001
        self.critic_lr = 0.0001

        self.actor_net = ActorNet1(state_dim, action_dim).to(device)  # 策略网络
        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=self.actor_lr)

        self.critic_net = CriticNet(state_dim).to(device)  # 价值网络
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

        # 目标的当前时刻的state_value
        td_target = reward_batch.unsqueeze(1) + gamma * self.critic_net(next_state_batch) * (
            (1 - done_batch).unsqueeze(1))

        # td_target = reward_batch.unsqueeze(1) + gamma * self.critic_delay_net(next_state_batch) * ((1 - done_batch).unsqueeze(1))

        # 时序差分的误差计算 目标的state_value与预测的state_value之差
        td_error = td_target - td_value
        self.memory.update(idxs, td_error.detach().squeeze().tolist())  # update td error

        # 对每个状态对应得动作价值用log函数
        log_prob = torch.log(self.actor_net(state_batch).gather(1, action_batch))

        # 策略梯度损失
        actor_loss = torch.mean(-log_prob * td_value.detach())

        # 值函数损失，预测值与目标值之间
        critic_loss = torch.mean(F.mse_loss(self.critic_net(state_batch), td_target.detach()))

        self.actor_optimizer.zero_grad()  # 策略梯度网络的优化器
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()  # 价值网络的优化器
        critic_loss.backward()
        self.critic_optimizer.step()

        self.update_model()

        self.steps_done += 1

    def update_model(self):
        # for param, param_delay in zip(self.critic_net.parameters(), self.critic_delay_net.parameters()):
        #     value = param_delay.data *0.7 + param*0.3
        #     param_delay.data.copy_(value)
        pass


class Actor_2_Critic_Memory:
    def __init__(self, env):
        self.env = env
        self.epsilon = 1.0
        self.memory = ReplayMemory(10000)
        self.steps_done = 0
        self.name = 'Actor_2_Critic_Memory'
        self.actor_lr = 0.001
        self.critic_lr = 0.001

        self.actor_net = ActorNet1(state_dim, action_dim).to(device)  # 策略网络
        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=self.actor_lr)

        self.critic_net = CriticNet(state_dim).to(device)  # 价值网络
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
            action = action_list.sample().item()  # type : int

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

        # 预测的当前时刻的state_value
        td_value = self.critic_net(state_batch)
        # 目标的当前时刻的state_value
        td_target = reward_batch.unsqueeze(1) + gamma * self.critic_net(next_state_batch) * (
            (1 - done_batch).unsqueeze(1))
        # td_target = reward_batch.unsqueeze(1) + gamma * self.critic_delay_net(next_state_batch) * (
        #     (1 - done_batch).unsqueeze(1))
        # 时序差分的误差计算 目标的state_value与预测的state_value之差
        td_error = td_target - td_value  # tderror = r + γQ(s') - Q(s)

        # 对每个状态对应得动作价值用log函数
        log_prob = torch.log(self.actor_net(state_batch).gather(1, action_batch))
        # 策略梯度损失
        actor_loss = torch.mean(-log_prob * td_error.detach())
        # 值函数损失，预测值与目标值之间
        critic_loss = torch.mean(F.mse_loss(self.critic_net(state_batch), td_target.detach()))

        self.actor_optimizer.zero_grad()  # 策略梯度网络的优化器
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()  # 价值网络的优化器
        critic_loss.backward()
        self.critic_optimizer.step()



        self.update_model()
        self.steps_done += 1

    def update_model(self):
        # for param, param_delay in zip(self.critic_net.parameters(), self.critic_delay_net.parameters()):
        #     value = param_delay.data *0.7 + param*0.3
        #     param_delay.data.copy_(value)
        pass




class Actor_2_Critic_PreMemory:
    def __init__(self, env):
        self.env = env
        self.epsilon = 1.0
        self.memory = ReplayMemory_Per(10000)
        self.steps_done = 0
        self.name = 'Actor_2_Critic_PreMemory'
        self.actor_lr = 0.001
        self.critic_lr = 0.001

        self.actor_net = ActorNet1(state_dim, action_dim).to(device)  # 策略网络
        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=self.actor_lr)

        self.critic_net = CriticNet(state_dim).to(device)  # 价值网络
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
            action = action_list.sample().item()  # type : int

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
        # 目标的当前时刻的state_value
        td_target = reward_batch.unsqueeze(1) + gamma * self.critic_net(next_state_batch) * (
            (1 - done_batch).unsqueeze(1))
        # td_target = reward_batch.unsqueeze(1) + gamma * self.critic_delay_net(next_state_batch) * (
        #     (1 - done_batch).unsqueeze(1))
        # 时序差分的误差计算 目标的state_value与预测的state_value之差
        td_error = td_target - td_value  # tderror = r + γQ(s') - Q(s)
        self.memory.update(idxs, td_error.detach().squeeze().tolist())  # update td error
        # 对每个状态对应得动作价值用log函数
        log_prob = torch.log(self.actor_net(state_batch).gather(1, action_batch))
        # 策略梯度损失
        actor_loss = torch.mean(-log_prob * td_error.detach())
        # 值函数损失，预测值与目标值之间
        critic_loss = torch.mean(F.mse_loss(self.critic_net(state_batch), td_target.detach()))

        self.actor_optimizer.zero_grad()  # 策略梯度网络的优化器
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()  # 价值网络的优化器
        critic_loss.backward()
        self.critic_optimizer.step()


        self.update_model()
        self.steps_done += 1

    def update_model(self):
        # for param, param_delay in zip(self.critic_net.parameters(), self.critic_delay_net.parameters()):
        #     value = param_delay.data *0.7 + param*0.3
        #     param_delay.data.copy_(value)
        pass



'''
熵正则化
'''


class SAC:
    def __init__(self, env):
        self.env = env
        self.epsilon = 1.0
        self.memory = ReplayMemory(10000)
        self.steps_done = 0
        self.name = 'SAC'
        self.actor_lr = 0.01
        self.critic_lr = 0.01
        self.alpha_lr = 1e-4
        self.alpha = 0.036


        self.log_alpha = torch.tensor(np.log(self.alpha), dtype=torch.float)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr = self.actor_lr)


        self.target_entropy = -torch.tensor(np.log(action_dim))


        self.actor_net = ActorNet1(state_dim, action_dim).to(device)

        self.critic_net1 = CriticNet2(state_dim, action_dim).to(device)
        self.critic_net2 = CriticNet2(state_dim, action_dim).to(device)
        self.target_critic_net1 = CriticNet2(state_dim, action_dim).to(device)
        self.target_critic_net2 = CriticNet2(state_dim, action_dim).to(device)

        self.target_critic_net1.load_state_dict(self.critic_net1.state_dict())
        self.target_critic_net2.load_state_dict(self.critic_net2.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=self.actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_net1.parameters(), lr=self.critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_net2.parameters(), lr=self.critic_lr)

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
            action = action_list.sample().item()  # type : int

            return action

    def get_prob_entropy(self, state):  # 得到动作概率分布及  根据概率分布得到的熵    H(p) = E(-log(x)) = ∑ -log(x)
        with torch.no_grad():
            prob = self.actor_net(state)      # [64,26]
            if torch.isnan(prob).any():
                print(prob)
                print("NaN detected in prob!")

            entropy = prob * (prob + 1e-8).log()
            entropy = -entropy.sum(dim=1, keepdim=True)

        return prob, entropy


    def learn(self):
        if len(self.memory) < batch_size:
            return
        transitions = self.memory.sample(batch_size)
        batch = list(zip(*transitions))
        state_batch = torch.tensor(batch[0], dtype=torch.float32).to(device)   # [64,3]
        action_batch = torch.tensor(batch[1], dtype=torch.long).view(-1, 1).to(device)  # [64,1]
        reward_batch = torch.tensor(batch[2], dtype=torch.float32).to(device)  #[64]
        next_state_batch = torch.tensor(batch[3], dtype=torch.float32).to(device)  #[64,3]
        done_batch = torch.tensor(batch[4], dtype=torch.float32).to(device)   #[64]


        # 计算下一个动作的熵
        with torch.no_grad():
            prob_next_state, entropy_next_state = self.get_prob_entropy(next_state_batch)  # prob:[64,26]  entropy:[64,1]
            target1 = self.target_critic_net1(next_state_batch)
            target2 = self.target_critic_net2(next_state_batch)
            target = torch.min(target1, target2)                     # [64,26]



        # 更新 actor_net
        # 计算当前动作的熵及概率
        prob_state, entropy_state = self.get_prob_entropy(state_batch)
        # 更新actor_net
        td_actor_value1 = self.critic_net1(state_batch)
        td_actor_value2 = self.critic_net2(state_batch)
        td_actor_value = torch.min(td_actor_value1, td_actor_value2)

        # 期望和
        td_actor_value_sum = (prob_state * td_actor_value).sum(dim=1, keepdim=True)

        # 加权熵
        actor_loss = -(td_actor_value_sum + self.log_alpha.exp()* entropy_state).mean()
        # actor_loss = -(td_actor_value_sum + self.alpha* entropy_state).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()



        #  加权熵，越大越好
        entropy_value = (prob_next_state * target).sum(dim=1, keepdim=True)   # [64, 1]
        entropy_value = entropy_value + self.log_alpha.exp() * entropy_next_state  # [64, 1]
        # entropy_value = entropy_value + self.alpha * entropy_next_state  # [64, 1]
        entropy_value = entropy_value * gamma * ((1 - done_batch).unsqueeze(1)) + reward_batch.unsqueeze(1)  #[64,64]

        # 更新critic_net   (yi, r+entropy)
        # 计算当前时刻的state_value     critic_net1(state_batch) :[64,26]
        td_critic_value1 = self.critic_net1(state_batch).gather(1, action_batch)          # [64,1]
        critic1_loss = F.mse_loss(td_critic_value1, entropy_value)
        self.critic_1_optimizer.zero_grad()
        critic1_loss.backward(retain_graph = True)
        self.critic_1_optimizer.step()

        td_critic_value2 = self.critic_net2(state_batch).gather(1, action_batch)
        critic2_loss = F.mse_loss(td_critic_value2, entropy_value)
        self.critic_2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic_2_optimizer.step()

        # # 更新alpha值    E = -α（logΠ+H0）
        alpha_loss = torch.mean(self.log_alpha.exp() * (entropy_state - self.target_entropy).detach())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.update_model(self.critic_net1,self.target_critic_net1)
        self.update_model(self.critic_net2,self.target_critic_net2)
        self.steps_done += 1


    def update_model(self, _from, _to):
        if self.steps_done % 10 == 0:
            for _from, _to in zip(_from.parameters(), _to.parameters()):
                value = _to.data * 0.95 + _from.data * 0.05
                _to.data.copy_(value)



######################
class SAC_Prememory:
    def __init__(self, env):
        self.env = env
        self.epsilon = 1.0
        self.memory = ReplayMemory_Per(10000)
        self.steps_done = 0
        self.name = 'SAC_Prememory'
        self.actor_lr = 0.01
        self.critic_lr = 0.01
        self.alpha_lr = 1e-4
        self.alpha = 0.036


        self.log_alpha = torch.tensor(np.log(self.alpha), dtype=torch.float)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr = self.actor_lr)


        self.target_entropy = -torch.tensor(np.log(action_dim))


        self.actor_net = ActorNet1(state_dim, action_dim).to(device)

        self.critic_net1 = CriticNet2(state_dim, action_dim).to(device)
        self.critic_net2 = CriticNet2(state_dim, action_dim).to(device)
        self.target_critic_net1 = CriticNet2(state_dim, action_dim).to(device)
        self.target_critic_net2 = CriticNet2(state_dim, action_dim).to(device)

        self.target_critic_net1.load_state_dict(self.critic_net1.state_dict())
        self.target_critic_net2.load_state_dict(self.critic_net2.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=self.actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_net1.parameters(), lr=self.critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_net2.parameters(), lr=self.critic_lr)

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
            action = action_list.sample().item()  # type : int

            return action

    def get_prob_entropy(self, state):  # 得到动作概率分布及  根据概率分布得到的熵    H(p) = E(-log(x)) = ∑ -log(x)
        with torch.no_grad():
            prob = self.actor_net(state)      # [64,26]
            if torch.isnan(prob).any():
                print(prob)
                print("NaN detected in prob!")

            entropy = prob * (prob + 1e-8).log()
            entropy = -entropy.sum(dim=1, keepdim=True)

        return prob, entropy


    def learn(self):
        if self.memory.size() < batch_size:
            return
        idxs, transitions = self.memory.sample(batch_size)
        batch = list(zip(*transitions))
        state_batch = torch.tensor(batch[0], dtype=torch.float32).to(device)   # [64,3]
        action_batch = torch.tensor(batch[1], dtype=torch.long).view(-1, 1).to(device)  # [64,1]
        reward_batch = torch.tensor(batch[2], dtype=torch.float32).to(device)  #[64]
        next_state_batch = torch.tensor(batch[3], dtype=torch.float32).to(device)  #[64,3]
        done_batch = torch.tensor(batch[4], dtype=torch.float32).to(device)   #[64]


        # 计算下一个动作的熵
        with torch.no_grad():
            prob_next_state, entropy_next_state = self.get_prob_entropy(next_state_batch)  # prob:[64,26]  entropy:[64,1]
            target1 = self.target_critic_net1(next_state_batch)
            target2 = self.target_critic_net2(next_state_batch)
            target = torch.min(target1, target2)                     # [64,26]



        # 更新 actor_net
        # 计算当前动作的熵及概率
        prob_state, entropy_state = self.get_prob_entropy(state_batch)
        # 更新actor_net
        td_actor_value1 = self.critic_net1(state_batch)
        td_actor_value2 = self.critic_net2(state_batch)
        td_actor_value = torch.min(td_actor_value1, td_actor_value2)

        # 期望和
        td_actor_value_sum = (prob_state * td_actor_value).sum(dim=1, keepdim=True)

        # 加权熵
        actor_loss = -(td_actor_value_sum + self.log_alpha.exp()* entropy_state).mean()
        # actor_loss = -(td_actor_value_sum + self.alpha* entropy_state).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        #  加权熵，越大越好
        entropy_value = (prob_next_state * target).sum(dim=1, keepdim=True)   # [64, 1]
        entropy_value = entropy_value + self.log_alpha.exp() * entropy_next_state  # [64, 1]
        # entropy_value = entropy_value + self.alpha * entropy_next_state  # [64, 1]
        entropy_value = entropy_value * gamma * ((1 - done_batch).unsqueeze(1)) + reward_batch.unsqueeze(1)  #[64,64]

        # 更新critic_net   (yi, r+entropy)
        # 计算当前时刻的state_value     critic_net1(state_batch) :[64,26]
        td_critic_value1 = self.critic_net1(state_batch).gather(1, action_batch)
        td_error1 = td_critic_value1-entropy_value# [64,1]
        critic1_loss = F.mse_loss(td_critic_value1, entropy_value)
        self.critic_1_optimizer.zero_grad()
        critic1_loss.backward(retain_graph=True)
        self.critic_1_optimizer.step()



        td_critic_value2 = self.critic_net2(state_batch).gather(1, action_batch)
        td_error2 = td_critic_value2 - entropy_value
        critic2_loss = F.mse_loss(td_critic_value2, entropy_value)
        self.critic_2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic_2_optimizer.step()


        td_error = torch.min(td_error1,td_error2)
        self.memory.update(idxs, td_error.detach().squeeze().tolist())


        # # 更新alpha值    E = -α（logΠ+H0）
        alpha_loss = torch.mean(self.log_alpha.exp() * (entropy_state - self.target_entropy).detach())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.update_model(self.critic_net1,self.target_critic_net1)
        self.update_model(self.critic_net2,self.target_critic_net2)
        self.steps_done += 1


    def update_model(self, _from, _to):
        if self.steps_done % 10 == 0:
            for _from, _to in zip(_from.parameters(), _to.parameters()):
                value = _to.data * 0.95 + _from.data * 0.05
                _to.data.copy_(value)
