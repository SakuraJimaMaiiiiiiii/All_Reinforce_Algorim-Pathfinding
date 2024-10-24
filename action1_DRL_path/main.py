from environment import Environment

from train import train

from utils import plot_3d_path, calculate_path_length, plot_results

from AC_Series_agent import Actor_Critic_Memory, Actor_Critic_PreMemory, Actor_2_Critic_Memory, \
    Actor_2_Critic_PreMemory, \
    SAC, SAC_Prememory

from Dqn_Serise_agent import DQN, DoubleDQN, DuelingQN, DuelingQN_Prememory
from PolicyGradient_Series_agent import reinforce


env = Environment()

''''
模型选择
'''
agent = DuelingQN_Prememory(env)
# agent = DuelingQN(env)
# agent = DoubleDQN(env)
# agent = DQN(env)




# agent = Actor_Critic_Memory(env)
# agent = SAC(env)
# agent = SAC_Prememory(env)
# agent = Actor_2_Critic_PreMemory(env)
# agent = Actor_2_Critic_Memory(env)
# agent = Actor_Critic_Memory(env)


# agent = reinforce(env)





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
