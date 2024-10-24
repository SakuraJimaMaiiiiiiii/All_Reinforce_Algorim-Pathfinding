import torch
from model import ActorNet1


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
EPSILON_END = 0.01
EPSILON_START = 1.0
EPSILON_DECAY = 0.95
learn_rate = 0.003
gamma = 0.9  # 折扣因子

state_dim = 3
action_dim = 26



class reinforce:
    def __init__(self,env):
        self.env = env
        self.steps_done = 0
        self.name = 'reinforce'
        self.policy_lr = 1e-3

        self.policy_net = ActorNet1(state_dim,action_dim).to(device)
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.policy_lr)

        self.transition_dict = {
                'state': [],
                'action': [],
                'reward': [],
                'next_state': [],
                'done': []
            }

    def store_transition(self, state, action, reward, next_stata, done):
        self.transition_dict['state'].append(state)
        self.transition_dict['action'].append(action)
        self.transition_dict['reward'].append(reward)
        self.transition_dict['next_state'].append(next_stata)
        self.transition_dict['done'].append(done)


    def choose_action(self,state):
        state = torch.tensor([state],dtype=torch.float).to(device)
        probs = self.policy_net(state)
        action_dist = torch.distributions.Categorical(probs)
        action =action_dist.sample()
        return action.item()


    def learn(self):
        stata_list = self.transition_dict['state']
        reward_list = self.transition_dict['reward']
        action_list = self.transition_dict['action']


        reward_sum = 0
        self.policy_optimizer.zero_grad()
        for i in reversed(range(len(reward_list))):
            reward = reward_list[i]
            stata = torch.tensor([stata_list[i]],dtype=torch.float).to(device)
            action = torch.tensor([action_list[i]]).view(-1,1).to(device)
            log_prob = torch.log(self.policy_net(stata).gather(1,action))
            reward_sum = gamma*reward_sum + reward
            loss = -log_prob* reward_sum
            loss.backward()
        self.policy_optimizer.step()


    def update_model(self):
        pass


class PPO:
    def __init__(self,env):
        self.env = env