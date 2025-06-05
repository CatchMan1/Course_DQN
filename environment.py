import gym
import numpy as np
from gym import spaces

class UserSimEnv(gym.Env):
    def __init__(self, item_pool, user_simulator, max_steps=15):
        super().__init__()
        # 定义动作空间：一次从 候选集中选一个，action 就是 选出的候选课程的索引
        self.item_pool = item_pool

        # 定义状态空间，状态包括：用户简档，历史项目交互集，历史喜欢的项目，历史讨厌的项目
        # 获取一次状态向量的维度，用于定义 observation_space
        self.user_simulator = user_simulator #逻辑模型+统计模型
        # 每次推荐最大推荐的课程数
        self.max_steps = max_steps
        self._max_episode_steps = max_steps #重写时保持格式一致
        dummy_history = list(self.user_simulator.history)
        self.action_space = spaces.Discrete(len(item_pool))   # 动作空间是离散的，大小为候选课程池的长度
        print("action_space:", self.action_space)
        self.history = dummy_history
        dummy_state = self._make_state()
        obs_dim = dummy_state.shape[0]# 获取实时维度
        # self.K = 3 #使用最近的K次历史交互课程
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        """采用与版本相同的调用接口"""
        super().reset(seed=seed)  # 调用 gym.Env 的基础重置逻辑（设定随机种子）
        self.history = list(self.user_simulator.history)
        self.current_step = 0
        state = self._make_state()
        info = {
            "user_id": self.user_simulator.user_id,
            "init_profile": self.user_simulator.user_profile_emb.tolist(),
            "init_history": list(self.history),
        }
        return state, info

    def step(self, action):
        # action 是一个整数，代表 item_pool[action]
        item = self.item_pool[action] # action为取出项目的索引，取出的为candidate_item
        item_index = action
        # print("item_index:", item_index)

        # 使用 两个逻辑模型 + 统计模型 模拟器产生反馈 0/1
        feedback = self.user_simulator.predict_feedback(self.history, item_index)
        # 设置奖励
        reward = 1 if feedback == 1 else 0

        # 更新历史状态
        self.history.append((item_index, feedback))
        self.current_step += 1
        next_state = self._make_state()
        terminated = False
        truncated = (self.current_step >= self.max_steps)
        info = {
        "user_id": self.user_simulator.user_id,
        "feedback": feedback,
        "current_item": item_index,
        "step": self.current_step
        }
        # print("recommendation info:", info)
        return next_state, reward, terminated, truncated, info  # 新版Gym返回五个值，把done改为terminated + truncated

    # 把用户简档、历史交互，Ipos、Ineg进行拼接
    def _make_state(self):
        # 用户简档嵌入
        user_emb = self.user_simulator.get_user_embedding()  # shape=(U,)
        D = self.user_simulator.item_emb_dim

        # 全部历史 embedding 池化
        item_ids = [item for item, _ in self.history]
        if item_ids:
            hist_emb = np.mean(
                [self.user_simulator.get_item_embedding(i) for i in item_ids],
                axis=0
            )
        else:
            hist_emb = np.zeros(D)

        # Ipos 全历史池化
        pos_ids = [i for i, fb in self.history if fb == 1]
        if pos_ids:
            pos_emb = np.mean(
                [self.user_simulator.get_item_embedding(i) for i in pos_ids],
                axis=0
            )
        else:
            pos_emb = np.zeros(D)

        # Ineg 全历史池化
        neg_ids = [i for i, fb in self.history if fb == 0]
        if neg_ids:
            neg_emb = np.mean(
                [self.user_simulator.get_item_embedding(i) for i in neg_ids],
                axis=0
            )
        else:
            neg_emb = np.zeros(D)

        # 拼接 state 向量
        state_vec = np.concatenate([user_emb, hist_emb, pos_emb, neg_emb], axis=0)
        return state_vec.astype(np.float32)


