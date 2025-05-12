import torch
import numpy as np
import gym
from torch.utils.tensorboard import SummaryWriter
from replay_buffer import *
from course_dqn import DQN
import argparse
from environment import UserSimEnv
from simulator import UserSimulator
from sentence_transformers import SentenceTransformer
from SAS_Rec.model import SASRec
from SAS_Rec_Wrapper import SASRecWrapper
class Runner:
    def __init__(self, args, user_simulator, number, seed):
        self.args = args
        self.user_simulator = user_simulator
        self.number = number
        self.seed = seed
        self.item_pool = self.user_simulator.item_pool

        self.env = UserSimEnv(item_pool, user_simulator)
        self.env_evaluate = UserSimEnv(item_pool, user_simulator)
        self.env.reset(seed=seed)# 训练重置环境
        self.env_evaluate.reset(seed=seed) # 评估重置环境
        np.random.seed(seed)
        torch.manual_seed(seed)# 固定模型初始化权重

        self.args.state_dim = self.env.observation_space.shape[0]# 返回环境状态空间的维度
        self.args.action_dim = self.env.action_space.n# 返回动作空间的维度
        self.args.episode_limit = self.env._max_episode_steps  # 最大推荐课程数
        print("state_dim={}".format(self.args.state_dim))
        print("action_dim={}".format(self.args.action_dim))
        print("episode_limit={}".format(self.args.episode_limit))

        if args.use_per and args.use_n_steps:
            self.replay_buffer = N_Steps_Prioritized_ReplayBuffer(args)
        elif args.use_per:
            self.replay_buffer = Prioritized_ReplayBuffer(args)
        elif args.use_n_steps:
            self.replay_buffer = N_Steps_ReplayBuffer(args)
        else:
            self.replay_buffer = ReplayBuffer(args)
        self.agent = DQN(args)

        self.algorithm = 'DQN'
        if args.use_double and args.use_dueling and args.use_noisy and args.use_per and args.use_n_steps:
            self.algorithm = 'Course_' + self.algorithm
        else:
            if args.use_double:
                self.algorithm += '_Double'
            if args.use_dueling:
                self.algorithm += '_Dueling'
            if args.use_noisy:
                self.algorithm += '_Noisy'
            if args.use_per:
                self.algorithm += '_PER'
            if args.use_n_steps:
                self.algorithm += "_N_steps"

        self.writer = SummaryWriter(log_dir='runs/DQN/{}_number_{}_seed_{}'.format(self.algorithm, number, seed))

        self.evaluate_num = 0  # Record the number of evaluations
        self.evaluate_rewards = []  # Record the rewards during the evaluating
        self.total_steps = 0  # Record the total steps during the training
        if args.use_noisy:  # 如果使用Noisy net，就不需要epsilon贪心策略了
            self.epsilon = 0
        else:
            self.epsilon = self.args.epsilon_init
            self.epsilon_min = self.args.epsilon_min
            self.epsilon_decay = (self.args.epsilon_init - self.args.epsilon_min) / self.args.epsilon_decay_steps

    def run(self, ):
        self.evaluate_policy()
        while self.total_steps < self.args.max_train_steps:
            state, _ = self.env.reset()
            done = False
            episode_steps = 0
            while not done:
                # ε–贪婪选动作：网络输入是 state，输出 len(item_pool) 个 Q 值
                action = self.agent.choose_action(state, epsilon=self.epsilon)

                # —— 这里用训练环境 self.env.step 而不是 self.env_evaluate
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                episode_steps += 1
                self.total_steps += 1

                # decay ε
                if not self.args.use_noisy:
                    self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_min)

                # gym 新 API 要区分 terminated vs truncated
                terminal_flag = terminated  # 如果你将来有自然结束条件，就用 terminated；否则一般都是 False

                # 存储 transition
                self.replay_buffer.store_transition(
                    state, action, reward, next_state, terminal_flag, done
                )
                state = next_state

                # DQN 学习
                if self.replay_buffer.current_size >= self.args.batch_size:
                    self.agent.learn(self.replay_buffer, self.total_steps)
            # 每隔 evaluate_freq 步再评估一次
            if self.total_steps % self.args.evaluate_freq == 0:
                self.evaluate_policy()

        # Save reward
        np.save('DRL-code-pytorch-main/Course_DQN/data_train/{}_number_{}_seed_{}.npy'.format(self.algorithm, self.number, self.seed), np.array(self.evaluate_rewards))

    def evaluate_policy(self):
        evaluate_reward = 0
        self.agent.net.eval()  # 切到评估模式，不开启 dropout/batchnorm 更新

        for _ in range(self.args.evaluate_times):
            # reset 会返回 (state, info)
            state, info = self.env_evaluate.reset()
            done = False
            episode_reward = 0

            # 你可以在 info 里拿到 user_id，如果想打印的话：
            # print(f"Evaluating for user {info['user_id']}")

            while not done:
                # ε=0 的贪婪，即始终选 Q 最大的动作
                action = self.agent.choose_action(state, epsilon=0)
                next_state, reward, terminated, truncated, step_info = self.env_evaluate.step(action)
                done = terminated or truncated

                episode_reward += reward
                state = next_state

            evaluate_reward += episode_reward

        self.agent.net.train()  # 恢复训练模式
        avg_reward = evaluate_reward / self.args.evaluate_times
        self.evaluate_rewards.append(avg_reward)

        print(f"total_steps:{self.total_steps}\t"
            f"evaluate_reward:{avg_reward:.3f}\t"
            f"epsilon:{self.epsilon:.3f}")
        self.writer.add_scalar(f'step_rewards_model', avg_reward, global_step=self.total_steps)




if __name__ == '__main__':
    # 初始化超参数
    parser = argparse.ArgumentParser("Hyperparameter Setting for DQN")
    parser.add_argument("--max_train_steps", type=int, default=int(4e5), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=1e3, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=float, default=3, help="Evaluate times")

    parser.add_argument("--buffer_capacity", type=int, default=int(1e5), help="The maximum replay-buffer capacity ")
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    parser.add_argument("--hidden_dim", type=int, default=256, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate of actor")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon_init", type=float, default=0.5, help="Initial epsilon")
    parser.add_argument("--epsilon_min", type=float, default=0.1, help="Minimum epsilon")
    parser.add_argument("--epsilon_decay_steps", type=int, default=int(1e5), help="How many steps before the epsilon decays to the minimum")
    parser.add_argument("--tau", type=float, default=0.005, help="soft update the target network")
    parser.add_argument("--use_soft_update", type=bool, default=True, help="Whether to use soft update")
    parser.add_argument("--target_update_freq", type=int, default=200, help="Update frequency of the target network(hard update)")
    parser.add_argument("--n_steps", type=int, default=5, help="n_steps")
    parser.add_argument("--alpha", type=float, default=0.6, help="PER parameter")
    parser.add_argument("--beta_init", type=float, default=0.4, help="Important sampling parameter in PER")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Learning rate Decay")
    parser.add_argument("--grad_clip", type=float, default=10.0, help="Gradient clip")

    parser.add_argument("--use_double", type=bool, default=True, help="Whether to use double Q-learning")
    parser.add_argument("--use_dueling", type=bool, default=True, help="Whether to use dueling network")
    parser.add_argument("--use_noisy", type=bool, default=True, help="Whether to use noisy network")
    # 采用哪种经验放回的方法
    parser.add_argument("--use_per", type=bool, default=True, help="Whether to use PER")
    parser.add_argument("--use_n_steps", type=bool, default=True, help="Whether to use n_steps Q-learning")
    # 为 SASRec 添加专用超参 #
    parser.add_argument("--num_blocks",   type=int,   default=2,    help="SASRec Transformer block 数量")
    parser.add_argument("--num_heads",    type=int,   default=1,    help="SASRec 多头注意力 head 数量")
    parser.add_argument("--hidden_units", type=int,   default=50,   help="SASRec 隐层维度")
    parser.add_argument("--maxlen",       type=int,   default=200,  help="SASRec 序列最大长度")
    parser.add_argument("--dropout_rate", type=float, default=0.2,  help="SASRec Dropout rate")
    # 其它 SASRec 里用到但你没定义的参数也一并加上：
    parser.add_argument("--lr_emb",       type=float, default=0.0,  help="SASRec L2 regularization for embeddings")

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    user_id = ["A", "B"]
    item_pool = ["人工智能","机器学习","R语言可视化","现代语言学"]
    item_keywords_pos = {
                "人工智能": {"自动化", "智能决策", "深度学习"},
                "机器学习": {"监督学习", "模型评估", "特征工程"},
                "R语言可视化": {"图表", "数据探索", "ggplot2"},
                "现代语言学": {"语法分析", "语义学", "语言变迁"}
            }
    item_keywords_neg = {
                "人工智能": {"高计算成本", "难以解释"},
                "机器学习": {"过拟合", "数据需求大"},
                "R语言可视化": {"学习曲线陡峭", "性能瓶颈"},
                "现代语言学": {"理论性强", "实践性弱"}
            }
     # 定义两个用户的历史交互示例
    history_user_A = [("机器学习", 1), ("现代语言学", 0)]
    history_user_B = [("人工智能", 1), ("R语言可视化", 1), ("机器学习", 0)]
     # 用户简档
    user_text = "用户偏好 深度学习 数据探索 自动化"
    # 具体要使用Bert序列（这里进行占位）
    # 设每个课程 embedding 维度为 8(根据实际修改)
    
    # 初始化Bert模型
    model = SentenceTransformer('all-MiniLM-L6-v2')  # 输出默认是 384 维向量

    #构造 item 的语义向量（拼接正 + 负关键词文本）
    item_embeddings = {
        item: model.encode(" ".join(item_keywords_pos[item] | item_keywords_neg[item]))
        for item in item_pool
    }
    # 用户简档 embedding 也为维度 8 的向量(暂时占位)，具体可能用其他嵌入
    user_profile_emb = model.encode(user_text)
    #user_profile_emb = np.random.rand(8)
    # 统计模型先设置为None
    # 假设 item_num = 100，device = 'cuda'
    '''
    device = args.device
    item2id = {item: idx + 1 for idx, item in enumerate(item_pool)}  # 保留 0 做 padding
    sasrec = SASRec(user_num=1, item_num=len(item2id), args=args)
    sasrec.load_state_dict(torch.load("DRL-code-pytorch-main/Course_DQN/SAS_Rec/ml-1m_default/SASRec.epoch=1000.lr=0.001.layer=2.head=1.hidden=50.maxlen=200.pth"))
    sasrec.to(device)

    # 3. 构造 wrapper
    stat_model = SASRecWrapper(sasrec_model=sasrec, item2id=item2id, maxlen=args.maxlen)
    
    '''
    stat_model = None
    # 用户 A
    sim_A = UserSimulator(
                user_id="user_A",
                item_pool=item_pool,
                item_keywords_pos=item_keywords_pos,
                item_keywords_neg=item_keywords_neg,
                item_embeddings=item_embeddings,
                user_profile_emb=user_profile_emb,
                init_history=history_user_A,
                statistical_model=stat_model
            )

    # 用户 B
    sim_B = UserSimulator(
                user_id="user_B",
                item_pool=item_pool,
                item_keywords_pos=item_keywords_pos,
                item_keywords_neg=item_keywords_neg,
                item_embeddings=item_embeddings,
                user_profile_emb=user_profile_emb,
                init_history=history_user_B,
                statistical_model=None
            )
#选择sim_A用户作为尝试
    for seed in [0, 10, 100]:
        runner = Runner(args=args, user_simulator=sim_A, number=1, seed=seed)# 选用CartPole-v1环境
        runner.run()
