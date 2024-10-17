import random
import numpy as np
from random import uniform


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    # def push(self, transition):
    #     self.memory.append(transition)
    #     if len(self.memory) > self.capacity:
    #         del self.memory[0]

    def push(self, transition):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity
        # print(transition)



    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

''''''''


class ReplayMemory_Per:
    # stored as ( s, a, r, s_ ) in SumTree
    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.memory_size = capacity
        self.prio_max = 0.1  # 记录当前样本的最大优先级 初始值为0.1 确保新经验有较高的采样优先级
        self.a = 0.6  # 优先级采样的权重系数 当 a=0 时，所有样本被均匀采样；a越大优先回访机制越强 a=1时采样完全按优先级比例进行  P(i) = pi^a/Σpk^a
        self.e = 0.01  # 一条(s,a,r,s')的TD-error δ = r +γQtarget(s',argmaxQ(s',a‘))-Q(s,a)

    # 存储经验
    def push(self, transition):
        data = transition  # Proportional prioritization: pi = |δi|+ e  保证 pi>0
        p = (np.abs(self.prio_max) + self.e) ** self.a  # 新采样的数据默认优先级最高，确保最近一次学习尽可能被抽样到 学习的时候再根据TD-error更新p
        self.tree.add(p, data)


    def sample(self, batch_size):
        idxs = []                 # 存储抽样到的经验的索引  batchsize个
        segment = self.tree.total() / batch_size  # self.tree.total()返回 最高节点的p值 分成batchsize个区间 n = sum(p) / batchsize
        sample_datas = []     # 存储抽样的数据池 batchsize个

        for i in range(batch_size):    # 从每个区间中抽样s
            a = segment * i
            b = segment * (i + 1)
            s = uniform(a, b)
            idx, p, data = self.tree.get(s)

            sample_datas.append(data)
            idxs.append(idx)
        return idxs, sample_datas

    def update(self, idxs, errors):   # errors:TD误差列表
        self.prio_max = max(self.prio_max, max(np.abs(errors)))
        for i, idx in enumerate(idxs):
            p = (np.abs(errors[i]) + self.e) ** self.a                # pi = |δi|+ e
            self.tree.update(idx, p)

    def size(self):
        return self.tree.n_entries



class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)  # 只储存叶节点的数据
        self.n_entries = 0

    # update to the root node

    # 递归
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2  # 根据子节点找到父节点

        self.tree[parent] += change

        if parent != 0:  # 将change值一直传递到最高点
            self._propagate(parent, change)

    # 递归
    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1  # 根据父节点找出子节点
        right = left + 1

        if left >= len(self.tree):  # 判断idx是否是叶节点 如果是叶节点 返回该idx
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)  # 一直找到子节点
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):  # p：新写入数据的优先级  data：实际存入的数据
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):              # 返回 (索引 对应树的值 数据)
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])