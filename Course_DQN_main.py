import torch
import numpy as np
# import gym
from torch.utils.tensorboard import SummaryWriter
from replay_buffer import *
from course_dqn import DQN
import argparse
from environment import UserSimEnv
from simulator import UserSimulator
from sentence_transformers import SentenceTransformer
from SASREC.Rec import load_plm, generate_item_embedding
from SASREC.A_SASRec_final_bce_llm import SASRec
import csv
import os
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
        self.agent = DQN(args)  # 初始化 DQN agent

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
                recommended_items = [h[0] for h in self.env.history]
                # 2. 生成可选动作（未推荐过的 item 的索引）
                available_actions = [item_id for item_id in self.item_pool if item_id not in recommended_items]
                # ε–贪婪选动作：网络输入是 state，输出 len(item_pool) 个 Q 值
                
                action = self.agent.choose_action(state, epsilon=self.epsilon, available_actions=available_actions)
    
                # 这里用训练环境 self.env.step 而不是 self.env_evaluate
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
                action_index = action - 1
                self.replay_buffer.store_transition(
                    state, action_index, reward, next_state, terminal_flag, done
                )
                state = next_state

                # DQN 学习
                if self.replay_buffer.current_size >= self.args.batch_size:
                    self.agent.learn(self.replay_buffer, self.total_steps)
            # 每隔 evaluate_freq 步再评估一次
            if self.total_steps % self.args.evaluate_freq == 0:
                self.evaluate_policy()

        # 保存模型
        save_dir = f'DRL-code-pytorch-main/Course_DQN/model/{self.user_simulator.user_id}'
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.agent.net.state_dict(),
               os.path.join(save_dir, f'{self.algorithm}_number_{self.number}_seed_{self.seed}.pth'))
        # 保存奖励
        save_dir_data = f'DRL-code-pytorch-main/Course_DQN/data_train/{self.user_simulator.user_id}'
        os.makedirs(save_dir_data, exist_ok=True)
        # np.save('DRL-code-pytorch-main/Course_DQN/data_train/{}_number_{}_seed_{}.npy'.format(self.algorithm, self.number, self.seed), np.array(self.evaluate_rewards))
        np.save(os.path.join(save_dir_data, '{}_number_{}_seed_{}.npy'.format(self.algorithm, self.number, self.seed)), np.array(self.evaluate_rewards))

    def evaluate_policy(self):
        evaluate_reward = 0
        self.agent.net.eval()  # 切到评估模式，不开启 dropout/batchnorm 更新
        # 新增：用于保存多个评估 episode 的动作序列和反馈
        all_action_sequences = []
        all_feedback_distributions = []

        for _ in range(self.args.evaluate_times):
            # reset 会返回 (state, info)
            state, info = self.env_evaluate.reset()
            done = False
            episode_reward = 0
            episode_actions = []
            episode_feedback = []

            while not done:
                recommended_items = [h[0] for h in self.env.history]
                # 生成可选动作（未推荐过的 item 的索引）
                available_actions = [item_id for item_id in self.item_pool if item_id not in recommended_items]
                # ε–贪婪选动作：网络输入是 state，输出 len(item_pool) 个 Q 值

                # ε=0 的贪婪，即始终选 Q 最大的动作
                action = self.agent.choose_action(state, epsilon=0, available_actions=available_actions)
                # print("evaluate action:", action)
                next_state, reward, terminated, truncated, step_info = self.env_evaluate.step(action)
                done = terminated or truncated

                episode_reward += reward
                state = next_state

                # 记录动作和反馈
                episode_actions.append(action)
                episode_feedback.append(step_info)

            evaluate_reward += episode_reward
            all_action_sequences.append(episode_actions)
            all_feedback_distributions.append(episode_feedback)

        self.agent.net.train()  # 恢复训练模式
        avg_reward = evaluate_reward / self.args.evaluate_times
        self.evaluate_rewards.append(avg_reward)

        print(f"total_steps:{self.total_steps}\t"
            f"evaluate_reward:{avg_reward:.3f}\t"
            f"epsilon:{self.epsilon:.3f}")
        self.writer.add_scalar(f'step_rewards_model', avg_reward, global_step=self.total_steps)
        # 保存动作序列和反馈分布
        save_dir_data = f'DRL-code-pytorch-main/Course_DQN/data_train/{self.user_simulator.user_id}'
        os.makedirs(save_dir_data, exist_ok=True)
        np.save(os.path.join(save_dir_data, f'{self.algorithm}_actions_number_{self.number}_seed_{self.seed}.npy'), np.array(all_action_sequences, dtype=object))
        np.save(os.path.join(save_dir_data, f'{self.algorithm}_feedbacks_number_{self.number}_seed_{self.seed}.npy'), np.array(all_feedback_distributions, dtype=object))
        # np.save(f'DRL-code-pytorch-main/Course_DQN/data_train/{self.algorithm}_actions_number_{self.number}_seed_{self.seed}.npy', np.array(all_action_sequences, dtype=object))
        # np.save(f'DRL-code-pytorch-main/Course_DQN/data_train/{self.algorithm}_feedbacks_number_{self.number}_seed_{self.seed}.npy', np.array(all_feedback_distributions, dtype=object))



if __name__ == '__main__':
    # 初始化超参数
    parser = argparse.ArgumentParser("Hyperparameter Setting for DQN")
    parser.add_argument("--max_train_steps", type=int, default=int(4e4), help=" Maximum number of training steps")
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
    parser.add_argument("--hidden_units", type=int,   default=768,  help="SASRec 隐层维度")  # 改为768
    parser.add_argument("--maxlen",       type=int,   default=200,  help="SASRec 序列最大长度")
    parser.add_argument("--dropout_rate", type=float, default=0.1,  help="SASRec Dropout rate")  # 和模型初始化一致
    parser.add_argument("--lr_emb",       type=float, default=0.0,  help="SASRec L2 regularization for embeddings")
    
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    user_ids = []
    user_profiles = {}
    item_pool = {}
    # 所有用户以及用户简档
    with open('DRL-code-pytorch-main/Course_DQN/SASREC/ai_user_summary_output_fixed.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过表头
        for row in reader:
            user_id = row[0]
            user_ids.append(user_id)
            user_profile = row[1]
            user_profiles[user_id] = user_profile
            item1_text = row[2]
            item2_text = row[3]
            item3_text = row[4]
            item4_text = row[5]
            item5_text = row[6]
            for item_text in [item1_text, item2_text, item3_text, item4_text, item5_text]:
                item_id = len(item_pool) + 1
                item_pool[item_id] = item_text

    user_id = user_ids[1] # 选择一个用户ID
    user_text = [user_profiles.get(user_id)] # 获取用户简档文本

    item_keywords_pos = {}
    item_keywords_neg = {}
    history_item = []
    with open(f'DRL-code-pytorch-main/Course_DQN/student_info/{user_id}.csv', 'r', encoding='utf-8') as f1:
        reader = csv.reader(f1)
        next(reader)  # 跳过表头
        for row in reader:
            item_id = int(row[0])
            item_name = row[1]
            history_item.append((item_id, int(row[4])))  
    with open('DRL-code-pytorch-main/Course_DQN/class_index.csv','r',encoding='utf-8') as f2:
        reader = csv.reader(f2)
        next(reader)
        for row in reader:
            item_id = int(row[0])
            item_keywords_pos[item_id] = set(row[3].split())
            item_keywords_neg[item_id] = set(row[4].split())

    device = args.device
    
    plm_tokenizer, plm_model = load_plm('bert-base-uncased')
    plm_model = plm_model.to(device)


    item_text_dic = {}
    # 理论上来说要有所有在item_pool中的项目的积极和消极关键词文本，
    # 但由于没有所有项目积极消极的数据，先用item_pool代替
    # #构造 item 的语义向量（拼接正 + 负关键词文本）
    item_text_dic = {
        item_id: " ".join(item_keywords_pos.get(item_id, set()) | item_keywords_neg.get(item_id, set()))
        for item_id in item_pool
}
    item_text_dic[0] = ""  # 添加 padding 项目，用空字符串
    # 批量生成item_embeddings 维度为（81，768）的tensor, 内容包含了课程的积极和消极关键词文本的嵌入
    item_embeddings = generate_item_embedding(item_text_dic, plm_tokenizer, plm_model, word_drop_ratio=-1)
    item_embeddings = torch.tensor(item_embeddings, dtype=torch.float32).to(device)
    with torch.no_grad():
        encoded = plm_tokenizer(user_text, padding=True, max_length=512, truncation=True, return_tensors='pt').to(device)
        outputs = plm_model(**encoded)
        masked_output = outputs.last_hidden_state * encoded['attention_mask'].unsqueeze(-1)
        mean_output = masked_output[:, 1:, :].sum(dim=1) / encoded['attention_mask'][:, 1:].sum(dim=-1, keepdim=True)
        user_profile_emb = mean_output[0].cpu().numpy()  # 如果只有一个用户，取第一个

    stat_model = SASRec(
        hidden_size=args.hidden_units,
        item_num=len(item_pool)+1,  # +1 是因为有 padding 项目
        state_size=args.hidden_units,
        dropout=args.dropout_rate,
        device=device,
        num_heads=args.num_heads,
        pre_embeddings=item_embeddings  # 传入张量形式的项目嵌入
        )  # +1是因为有padding
    stat_model.load_state_dict(torch.load("D:\Visual Studio Code\DRL-code-pytorch-main\Course_DQN\SASREC\SASRec_model.pth", map_location=device))
    stat_model.to(device)
    stat_model.eval()
    # 用户 A
    sim_A = UserSimulator(
                user_id=user_id,
                item_pool=item_pool,
                item_keywords_pos=item_keywords_pos,
                item_keywords_neg=item_keywords_neg,
                item_embeddings=item_embeddings,
                user_profile_emb=user_profile_emb,
                init_history=history_item,
                statistical_model=stat_model
            )

#选择sim_A用户作为尝试
    for seed in [0, 10, 100]:
        runner = Runner(args=args, user_simulator=sim_A, number=1, seed=seed)
        runner.run()
