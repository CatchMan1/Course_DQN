import os
import numpy as np
import matplotlib.pyplot as plt
import csv
from collections import defaultdict
item_pool = []
    # 所有用户以及用户简档
with open('DRL-code-pytorch-main/Course_DQN/SASREC/ai_user_summary_output_fixed.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)  # 跳过表头
    for row in reader:
        user_id = row[0]
        user_profile = row[1]
        item1_text = row[2]
        item2_text = row[3]
        item3_text = row[4]
        item4_text = row[5]
        item5_text = row[6]
        for item_text in [item1_text, item2_text, item3_text, item4_text, item5_text]:
            item_id = len(item_pool) + 1
            item_pool.append(item_id)
# Configuration 
ALGO       = 'Course_DQN'      # Runner.algorithm
NUMBER     = 1                 # Runner.number
SEEDS      = [0]      # 训练时用到的随机种子列表
EVALUATE_FREQ = 1000           # 评估频率（与你训练脚本里的一致）

# 所有文件都在这个目录下（根据自己项目结构修改）
DATA_DIR = 'DRL-code-pytorch-main/Course_DQN/data_train'

# 课程列表，对应动作索引

# 学习曲线 (Learning Curve)
def load_and_smooth_rewards(path):
    raw = np.load(path)
    smooth = []
    for i, r in enumerate(raw):
        if i == 0:
            smooth.append(r)
        else:
            smooth.append(0.9 * smooth[-1] + 0.1 * r) #采用了指数加权平均平滑处理
    return np.array(smooth)

plt.figure(figsize=(8,5))
for seed in SEEDS:
    rfile = f"{ALGO}_number_{NUMBER}_seed_{seed}.npy"
    path = os.path.join(DATA_DIR, rfile)
    rewards = load_and_smooth_rewards(path)
    # print("Rewards for seed {}: {}".format(seed, rewards))
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

# 参数
seed = 0

# 加载数据
feedbacks_path = os.path.join(DATA_DIR, f"{ALGO}_feedbacks_number_{NUMBER}_seed_{seed}.npy")
feedbacks = np.load(feedbacks_path, allow_pickle=True)
# 累计计数器
cumulative_likes = 0
cumulative_total = 0
# print("feedbacks:", feedbacks)
step_accuracy = dict()

# 按照 step 顺序遍历所有 episode
max_step = max([record['step'] for episode in feedbacks for record in episode])

# 统计 step -> 所有 feedback（跨 episode）
step_feedbacks = defaultdict(list)
for episode in feedbacks:
    for record in episode:
        step = record['step']
        step_feedbacks[step].append(record['feedback'])
print("step_feedbacks:", step_feedbacks)
# 按 step 计算累计准确率
accuracies = []
cumulative_likes = 0
cumulative_total = 0

steps = sorted(step_feedbacks.keys())
for step in steps:
    feedbacks_in_step = step_feedbacks[step]
    cumulative_likes += sum(feedbacks_in_step)
    cumulative_total += len(feedbacks_in_step)
    acc = cumulative_likes / cumulative_total
    accuracies.append(acc)

# 可视化
plt.figure(figsize=(8, 5))
plt.plot(steps, accuracies, marker='o', color='darkorange')
plt.xlabel('Step (Evaluation Round)', fontsize=12)
plt.ylabel('Cumulative Accuracy (Likes / Total)', fontsize=12)
plt.title(f'Cumulative Accuracy over Steps (Seed {seed})', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
