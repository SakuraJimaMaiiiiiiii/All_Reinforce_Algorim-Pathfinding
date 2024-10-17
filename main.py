from environment import Environment
from agent import DuelingQN, PerDuelingQN, DQN, DoubleDQN, Actor_Critic, Actor_2_Critic
from train import train
from utils import plot_3d_path, calculate_path_length, plot_results


env = Environment()


''''
模型选择
'''
# agent = PerDuelingQN(env)
# agent = DuelingQN(env)
# agent = DoubleDQN(env)
# agent = DQN(env)
# agent = Actor_Critic(env)
agent = Actor_2_Critic(env)

episode_durations, scores, steps_list = train(agent)

# final_path = []
# state = env.reset()
# final_path.append(state)
#
# for _ in range(3000):
#     action = agent.select_action(state)
#     next_state, reward, done = env.step(action)
#     final_path.append(next_state)
#     state = next_state
#     if done:
#         break



