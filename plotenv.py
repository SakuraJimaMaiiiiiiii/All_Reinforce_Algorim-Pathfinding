# import numpy as np
# import matplotlib
# matplotlib.use('TkAgg')
#
# import matplotlib.pyplot as plt
#
#
#
# obstacles = [
#     ((0, 0, 0), (3, 30, 5)),
#
#     ((0, 0, 5), (3, 5, 14)),
#     ((0, 5, 5), (3, 10, 20)),
#     ((0, 10, 5), (3, 15, 25)),
#     ((0, 15, 5), (3, 20, 22)),
#     ((0, 20, 5), (3, 25, 17)),
#     ((0, 25, 5), (3, 30, 14)),
#
#
#     ((5, 12, 10), (7, 14, 20)),
#     ((5, 16, 10), (7, 18, 20)),
#     ((5, 14, 10), (7, 16, 12)),
#     ((5, 14, 18), (7, 16, 20)),
#
#
#
#     ((10, 0, 0), (20, 10, 30)),
#     ((10, 20, 0), (20, 30, 30)),
#     ((10, 10, 0), (20, 20, 10)),
#     ((10, 10, 20), (20, 20, 30))
# ]
#
# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')
#
# for (x1, y1, z1), (x2, y2, z2) in obstacles:
#     ax.bar3d(x1, y1, z1, x2 - x1, y2 - y1, z2 - z1, color='gray', alpha=0.5)
#
# plt.show()

import numpy as np
import plotly.graph_objs as go
import plotly.io as pio

# 障碍物数据
obstacles = [
    ((0, 0, 0), (3, 30, 5)),
    ((0, 0, 5), (3, 5, 14)),
    ((0, 5, 5), (3, 10, 20)),
    ((0, 10, 5), (3, 15, 25)),
    ((0, 15, 5), (3, 20, 22)),
    ((0, 20, 5), (3, 25, 17)),
    ((0, 25, 5), (3, 30, 14)),
    ((5, 12, 10), (7, 14, 20)),
    ((5, 16, 10), (7, 18, 20)),
    ((5, 14, 10), (7, 16, 12)),
    ((5, 14, 18), (7, 16, 20)),
    ((10, 0, 0), (20, 10, 30)),
    ((10, 20, 0), (20, 30, 30)),
    ((10, 10, 0), (20, 20, 10)),
    ((10, 10, 20), (20, 20, 30))
]

# 创建 Plotly 的 3D 立方体
fig_data = []
for (x1, y1, z1), (x2, y2, z2) in obstacles:
    # 计算每个立方体的顶点
    x = [x1, x2, x2, x1, x1, x2, x2, x1]
    y = [y1, y1, y2, y2, y1, y1, y2, y2]
    z = [z1, z1, z1, z1, z2, z2, z2, z2]

    # 定义每个立方体的六个面，并设置为灰色和透明度
    fig_data.append(go.Mesh3d(
        x=x,
        y=y,
        z=z,
        color='gray',  # 设置颜色为灰色
        opacity=1,  # 设置透明度为0.5
        alphahull=0
    ))

# 设置图形布局
layout = go.Layout(
    title='3D Obstacles Plot',
    scene=dict(
        xaxis_title='X Axis',
        yaxis_title='Y Axis',
        zaxis_title='Z Axis',
    ),
    showlegend=False
)

# 创建图形
fig = go.Figure(data=fig_data, layout=layout)

# 保存为 HTML 文件
pio.write_html(fig, '3d_obstacles_plot.html')

print("3D 障碍物图已保存为 3d_obstacles_plot.html")
