import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import plotly.graph_objs as go
import plotly.io as pio
import torch


# 图像保存地址
pic_Duration_dir = r'E:\files\code\强化学习\D3QN\results\pic_Duration'
pic_Score_dir = r'E:\files\code\强化学习\D3QN\results\pic_Score'
pic_Steptime_dir = r'E:\files\code\强化学习\D3QN\results\pic_steptime'
pic_3d_dir = r'E:\files\code\强化学习\D3QN\results\pic_3d路径'
pic_3d_html = r'E:\files\code\强化学习\D3QN\results\pic_3d_html'
os.makedirs(pic_Duration_dir, exist_ok=True)
os.makedirs(pic_Score_dir, exist_ok=True)
os.makedirs(pic_Steptime_dir, exist_ok=True)
os.makedirs(pic_3d_dir, exist_ok=True)
os.makedirs(pic_3d_html,exist_ok=True)

def plot_3d_path(path, obstacles, episode):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    path = np.array(path)
    x, y, z = zip(*path)

    for (x1, y1, z1), (x2, y2, z2) in obstacles:
        ax.bar3d(x1, y1, z1, x2 - x1, y2 - y1, z2 - z1, color='gray', alpha=1)

    ax.plot(path[:, 0], path[:, 1], path[:, 2], c='blue')
    ax.scatter(x[0], y[0], z[0], marker='o', color='g', label='Start')
    ax.scatter(x[-1], y[-1], z[-1], marker='o', color='r', label='Goal')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title('3D Path Taken by Agent')
    plt.savefig(f'{pic_3d_dir}/{episode+1}.png')


#绘制3d图
    fig_data = []
    for (x1, y1, z1), (x2, y2, z2) in obstacles:
        x_obstacle = [x1, x2, x2, x1, x1, x2, x2, x1]
        y_obstacle = [y1, y1, y2, y2, y1, y1, y2, y2]
        z_obstacle = [z1, z1, z1, z1, z2, z2, z2, z2]
        fig_data.append(go.Mesh3d(
            x=x_obstacle,
            y=y_obstacle,
            z=z_obstacle,
            color='gray',  # 灰色障碍物
            opacity=1.0,  # 完全不透明
            alphahull=0
        ))

    fig_data.append(go.Scatter3d(
        x=path[:, 0], y=path[:, 1], z=path[:, 2],
        mode='lines', line=dict(color='blue', width=4),
        name='Path'
    ))
    # 绘制起点和终点
    fig_data.append(go.Scatter3d(
        x=[x[0]], y=[y[0]], z=[z[0]],
        mode='markers', marker=dict(color='green', size=8),
        name='Start'
    ))
    fig_data.append(go.Scatter3d(
        x=[x[-1]], y=[y[-1]], z=[z[-1]],
        mode='markers', marker=dict(color='red', size=8),
        name='Goal'
    ))

    # 设置布局
    layout = go.Layout(
        title='3D Path Taken by Agent',
        scene=dict(
            xaxis_title='X Axis',
            yaxis_title='Y Axis',
            zaxis_title='Z Axis',
        ),
            showlegend=False
    )

    # 创建图形
    fig = go.Figure(data=fig_data, layout=layout)

    # 保存为 HTML 文件，保存在指定目录中
    file_path = f'{pic_3d_html}/{episode + 1}_3d_path.html'
    pio.write_html(fig, file_path)



def calculate_path_length(path):
    total_length = 0
    for i in range(1, len(path)):
        current_step = path[i]
        previous_step = path[i - 1]
        step_length = np.sqrt((current_step[0] - previous_step[0]) ** 2 +
                              (current_step[1] - previous_step[1]) ** 2 +
                              (current_step[2] - previous_step[2]) ** 2)
        total_length += step_length
    return total_length

def plot_results(episode_durations, scores, steps, episode):

    plt.figure()
    plt.plot(range(1, len(episode_durations) + 1),episode_durations)
    plt.title('Episode durations over time')
    plt.xlabel('Episode')
    plt.ylabel('Duration (seconds)')
    plt.savefig(f'{pic_Duration_dir}/{episode + 1}.png')
    plt.clf()


    plt.figure()
    plt.plot(range(1, len(scores) + 1),scores)
    plt.title('Scores over time')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.savefig(f'{pic_Score_dir}/{episode + 1}.png')
    plt.clf()


    plt.figure()
    plt.plot(range(1, len(steps) + 1),steps)
    plt.title('Steps per episode over time')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.tight_layout()
    plt.savefig(f'{pic_Steptime_dir}/{episode + 1}.png')
    plt.clf()



def load_models(self, episode):
    """
    loads the target actor and critic models, and copies them onto actor and critic models
    :param episode: the count of episodes iterated (used to find the file name)
    :return:
    """
    self.actor.load_state_dict(torch.load('./Models/' + str(episode) + '_actor.pt'))
    self.critic.load_state_dict(torch.load('./Models/' + str(episode) + '_critic.pt'))

    print('Models loaded succesfully')


# 添加噪声
class OrnsteinUhlenbeckActionNoise:

    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.X = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.X = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.X)
        dx = dx + self.sigma * np.random.randn(len(self.X))
        self.X = self.X + dx
        return self.X