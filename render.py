import os
import numpy as np
import matplotlib.pyplot as plt

# Configuration 
ALGO       = 'Course_DQN'      # Runner.algorithm
NUMBER     = 1                 # Runner.number
SEEDS      = [0, 10, 100]      # 训练时用到的随机种子列表
EVALUATE_FREQ = 1000           # 评估频率（与你训练脚本里的一致）

# 所有文件都在这个目录下（根据自己项目结构修改）
DATA_DIR = 'DRL-code-pytorch-main/Course_DQN/data_train'

# 课程列表，对应动作索引
item_pool = ["人工智能", "机器学习", "R语言可视化", "现代语言学"]

# 学习曲线 (Learning Curve)
def load_and_smooth_rewards(path):
    raw = np.load(path)
    smooth = []
    for i, r in enumerate(raw):
        if i == 0:
            smooth.append(r)
        else:
            smooth.append(0.9 * smooth[-1] + 0.1 * r)
    return np.array(smooth)

plt.figure(figsize=(8,5))
for seed in SEEDS:
    rfile = f"{ALGO}_number_{NUMBER}_seed_{seed}.npy"
    path = os.path.join(DATA_DIR, rfile)
    rewards = load_and_smooth_rewards(path)
    steps = np.arange(len(rewards)) * EVALUATE_FREQ
    plt.plot(steps, rewards, label=f'seed={seed}', alpha=0.7)

# 平均曲线
all_smooth = []
for seed in SEEDS:
    path = os.path.join(DATA_DIR, f"{ALGO}_number_{NUMBER}_seed_{seed}.npy")
    all_smooth.append(load_and_smooth_rewards(path))
all_smooth = np.stack(all_smooth, axis=0)
mean_rewards = all_smooth.mean(axis=0)
plt.plot(np.arange(len(mean_rewards)) * EVALUATE_FREQ,
         mean_rewards, color='black', linewidth=2, label='mean')

plt.title('Learning Curve')
plt.xlabel('Training Steps')
plt.ylabel('Average Eval Reward')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# #  推荐分布 (Recommendation Frequency)
# # 加载训练时记录的动作序列
# actions_path = os.path.join(DATA_DIR, f"{ALGO}_actions_{NUMBER}_{SEEDS[0]}.npy")
# # 这里以第一个 seed 为例，若要比较不同 seed，可多次绘制
# actions = np.load(actions_path)

# counts = np.bincount(actions, minlength=len(item_pool))
# plt.figure(figsize=(6,4))
# plt.bar(item_pool, counts, color='skyblue')
# plt.title('Recommendation Frequency')
# plt.xlabel('Course')
# plt.ylabel('Count')
# plt.xticks(rotation=30)
# plt.tight_layout()
# plt.show()


# # 反馈分布 (Feedback Distribution)
# feedbacks_path = os.path.join(DATA_DIR, f"{ALGO}_feedbacks_{NUMBER}_{SEEDS[0]}.npy")
# feedbacks = np.load(feedbacks_path)

# likes    = np.sum(feedbacks == 1)
# dislikes = np.sum(feedbacks == 0)
# plt.figure(figsize=(4,4))
# plt.pie(
#     [likes, dislikes],
#     labels=['Like','Dislike'],
#     colors=['lightgreen','salmon'],
#     autopct='%1.1f%%'
# )
# plt.title('Feedback Distribution')
# plt.tight_layout()
# plt.show()
