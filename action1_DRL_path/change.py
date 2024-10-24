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
