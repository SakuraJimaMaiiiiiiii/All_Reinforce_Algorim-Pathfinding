import time
import os
import torch
from utils import plot_results, plot_3d_path, calculate_path_length
from environment import Environment
from tqdm import tqdm


env = Environment()
NUM_EPISODES = 100
MAX_STEPS = 100

# 模型保存地址
policy_net_dir = r'E:\files\code\强化学习\D3QN\results\policymodel'
target_net_dir = r'E:\files\code\强化学习\D3QN\results\targetmodel'
actor_net_dir = r'E:\files\code\强化学习\D3QN\results\actormodel'
critic_net_dir =r'E:\files\code\强化学习\D3QN\results\criticmodel'
os.makedirs(actor_net_dir, exist_ok=True)
os.makedirs(critic_net_dir, exist_ok=True)
os.makedirs(policy_net_dir, exist_ok=True)
os.makedirs(target_net_dir, exist_ok=True)

def train(agent):
    episode_durations = []
    scores = []
    steps_list = []
    start_time = time.time()

    for episode in range(NUM_EPISODES):
        state = agent.env.reset()
        steps = 0
        score = 0


        phar = tqdm(total=MAX_STEPS)
        for t in range(MAX_STEPS):
            action = agent.choose_action(state)
            next_state, reward, done = agent.env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.learn()
            # agent.update_model()
            state = next_state
            score += reward
            steps += 1
            phar.update(1)

            if done or t == MAX_STEPS - 1:
                episode_durations.append(time.time() - start_time)
                scores.append(score)
                steps_list.append(steps)
                if (episode + 1) % 10 == 0:
                    path = get_path(env, agent)
                    path_len = calculate_path_length(path)
                    print(
                        f"Episode {episode + 1}, Steps: {steps}, Time: {episode_durations[-1]:.2f}s, Score: {score:.2f}, path_len:{path_len}")

                if (episode + 1) % 100 == 0:
                    if agent.name == 'Actor_2_Critic_PreMemory' or agent.name == 'Actor_Critic_PreMemory'or agent.name == 'Actor_Critic_Memory'or \
                            agent.name == 'Actor_2_Critic_Memory':
                        torch.save(agent.actor_net.state_dict(),f'{actor_net_dir}/actor_net_step_{episode + 1}.pth')
                        torch.save(agent.critic_net.state_dict(),f'{critic_net_dir}/critic_net_step_{episode + 1}.pth')

                    elif agent.name == 'SAC' or agent.name =='SAC_Prememory':
                        torch.save(agent.actor_net.state_dict(),f'{actor_net_dir}/actor_net_step_{episode + 1}.pth')
                        torch.save(agent.critic_net1.state_dict(),f'{critic_net_dir}/critic_net1_step_{episode + 1}.pth')
                        torch.save(agent.critic_net2.state_dict(),f'{critic_net_dir}/critic_net2_step_{episode + 1}.pth')
                        torch.save(agent.target_critic_net1.state_dict(),f'{critic_net_dir}/target_critic_net1_step_{episode + 1}.pth')
                        torch.save(agent.target_critic_net2.state_dict(),f'{critic_net_dir}/target_critic_net2_step_{episode + 1}.pth')

                    elif agent.name == 'reinforce':
                        pass
                    else:
                        torch.save(agent.policy_net.state_dict(), f'{policy_net_dir}/policy_net_step_{episode + 1}.pth')
                        torch.save(agent.target_net.state_dict(), f'{target_net_dir}/target_net_step_{episode + 1}.pth')

                    plot_results(episode_durations, scores, steps_list, episode)
                    path = get_path(env, agent)
                    plot_3d_path(path, env.obstacles, episode)
                break
    return episode_durations, scores, steps_list


def get_path(env, agent):
    final_path = []
    state = env.reset()
    final_path.append(state)
    step = 0
    # for _ in range(MAX_STEPS):
    #     action = agent.choose_action(state)
    #     next_state, reward, done = env.step(action)
    #     final_path.append(next_state)
    #     state = next_state
    #     if done:
    #         break
    while state != env.goal:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        final_path.append(next_state)
        state = next_state
        step += 1
        if step > MAX_STEPS:
            break
    return final_path
