import random

GRID_SIZE = 30
delta = 0.8  # 安全容量

class Environment:
    def __init__(self):
        self.grid_size = GRID_SIZE
        self.start = (0, GRID_SIZE//2, 0)
        self.goal = (GRID_SIZE - 1, GRID_SIZE // 2, GRID_SIZE - 1)
        #
        # self.obstacles = [
        #     ((3, 0, 0), (5, 2, 10)),
        #     ((3, 3, 0), (5, 5, 10)),
        #     ((3, 0, 10), (5, 5, 12)),
        #     ((10, 0, 0), (20, 10, 30)),
        #     ((10, 20, 0), (20, 30, 30)),
        #     ((10, 10, 0), (20, 20, 10)),
        #     ((10, 10, 20), (20, 20, 30))
        # ]

        self.obstacles = [
            ((1, 0, 0), (3, 30, 5)),

            ((1, 0, 5), (3, 5, 14)),
            ((1, 5, 5), (3, 10, 20)),
            ((1, 10, 5), (3, 15, 25)),
            ((1, 15, 5), (3, 20, 22)),
            ((1, 20, 5), (3, 25, 17)),
            ((1, 25, 5), (3, 30, 14)),

            ((5, 12, 10), (7, 14, 20)),
            ((5, 16, 10), (7, 18, 20)),
            ((5, 14, 10), (7, 16, 12)),
            ((5, 14, 18), (7, 16, 20)),

            ((10, 0, 0), (20, 10, 30)),
            ((10, 20, 0), (20, 30, 30)),
            ((10, 10, 0), (20, 20, 10)),
            ((10, 10, 20), (20, 20, 30))
        ]


    def is_in_obstacles(self, next_state):
        if (next_state[0] + delta, next_state[1]+delta, next_state[2]+delta) in self.obstacles:
            return True



    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        x, y, z = self.state
        if action == 0:  # Move up
            next_state = (x, min(y + 1, self.grid_size - 1), z)
            move_distance = 1
        elif action == 1:  # Move down
            next_state = (x, max(y - 1, 0), z)
            move_distance = 1

        elif action == 2:  # Move forward
            next_state = (min(x + 1, self.grid_size - 1), y, z)
            move_distance = 1

        elif action == 3:  # Move backward
            next_state = (max(x - 1, 0), y, z)
            move_distance = 1

        elif action == 4:  # Move left
            next_state = (x, y, max(z - 1, 0))
            move_distance = 1

        elif action == 5:  # Move right
            next_state = (x, y, min(z + 1, self.grid_size - 1))
            move_distance = 1
        elif action == 6:
            next_state = (min(x + 1, self.grid_size - 1), y, min(z + 1, self.grid_size - 1))
            move_distance = 1.4

        elif action == 7:
            next_state = (max(x - 1, 0), y, min(z + 1, self.grid_size - 1))
            move_distance = 1.4

        elif action == 8:
            next_state = (x, min(y + 1, self.grid_size - 1), min(z + 1, self.grid_size - 1))
            move_distance = 1.4

        elif action == 9:
            next_state = (x, max(y - 1, 0), min(z + 1, self.grid_size - 1))
            move_distance = 1.4

        elif action == 10:
            next_state = (min(x + 1, self.grid_size - 1), y, max(z - 1, 0))
            move_distance = 1.4

        elif action == 11:
            next_state = (max(x - 1, 0), y, max(z - 1, 0))
            move_distance = 1.4

        elif action == 12:
            next_state = (x, min(y + 1, self.grid_size - 1), max(z - 1, 0))
            move_distance = 1.4

        elif action == 13:
            next_state = (x, max(y - 1, 0), max(z - 1, 0))
            move_distance = 1.4

        elif action == 14:
            next_state = (min(x + 1, self.grid_size - 1), min(y + 1, self.grid_size - 1), z)
            move_distance = 1.4

        elif action == 15:
            next_state = (min(x + 1, self.grid_size - 1), max(y - 1, 0), z)
            move_distance = 1.4

        elif action == 16:  # Move right
            next_state = (max(x - 1, 0), min(y + 1, self.grid_size - 1), z)
            move_distance = 1.4

        elif action == 17:  # Move right
            next_state = (max(x - 1, 0), max(y - 1, 0), z)
            move_distance = 1.4

        # 一次动三格
        elif action == 18:
            next_state = (
                min(x + 1, self.grid_size - 1), min(y + 1, self.grid_size - 1), min(z + 1, self.grid_size - 1))
            move_distance = 1.7

        elif action == 19:
            next_state = (min(x + 1, self.grid_size - 1), max(y - 1, 0), min(z + 1, self.grid_size - 1))
            move_distance = 1.7

        elif action == 20:
            next_state = (max(x - 1, 0), min(y + 1, self.grid_size - 1), min(z + 1, self.grid_size - 1))
            move_distance = 1.7

        elif action == 21:
            next_state = (max(x - 1, 0), max(y - 1, 0), min(z + 1, self.grid_size - 1))
            move_distance = 1.7

        elif action == 22:
            next_state = (min(x + 1, self.grid_size - 1), min(y + 1, self.grid_size - 1), max(z - 1, 0))
            move_distance = 1.7

        elif action == 23:
            next_state = (min(x + 1, self.grid_size - 1), max(y - 1, 0), max(z - 1, 0))
            move_distance = 1.7

        elif action == 24:
            next_state = (max(x - 1, 0), min(y + 1, self.grid_size - 1), max(z - 1, 0))
            move_distance = 1.7

        elif action == 25:
            next_state = (max(x - 1, 0), max(y - 1, 0), max(z - 1, 0))
            move_distance = 1.7

        else:
            raise ValueError("Invalid action")


        # 检查是否在障碍或边界
        # if (next_state[0]+delta, next_state[1] + delta, next_state[2]+delta) in self.obstacles:
        if self.is_in_obstacles(next_state):
            reward = -1000
            # next_state = self.state  # Stay in the same place if obstacle encountered
            next_state = (x, y, z)

        elif next_state[0] <= (0 + delta) or next_state[0] >= (self.grid_size - delta) or \
                next_state[1] <= (0 + delta) or next_state[1] >= (self.grid_size - delta) or \
                next_state[2] <= (0 + delta) or next_state[2] >= (self.grid_size - delta):
            reward = -1000  # Penalty for going out of bounds
            # next_state = self.state  # Stay in the same place if out of bounds
            next_state = (x, y, z)

        else:
            reward = -move_distance


        # if next_state == self.goal:
        #     reward = 500
            # done = True

        # Check if reached the goal
        done = next_state == self.goal
        if done:
            reward = 5000

        self.state = next_state

        return next_state, reward, done

