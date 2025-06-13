import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
class UserSimulator:
  
    def __init__(self,user_id,item_pool,item_keywords_pos,item_keywords_neg,
                 item_embeddings,user_profile_emb, init_history,statistical_model=None):
        self.user_id = user_id
        self.item_pool = item_pool
        self.item_keywords_pos = item_keywords_pos
        self.item_keywords_neg = item_keywords_neg
        self.item_embeddings = item_embeddings
        self.user_profile_emb = user_profile_emb
        self.history = list(init_history) if init_history is not None else []# 保留副本，减少风险
        self.stat_model = statistical_model
        self.item_emb_dim = item_embeddings.shape[1]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.statistical_model = statistical_model.to(self.device) if statistical_model else None

    def get_user_embedding(self):
        
        return self.user_profile_emb

    def get_item_embedding(self, item_id):
        
        return self.item_embeddings[item_id].cpu().numpy()

    def f_mat(self, history, candidate_item):
        # Keyword matching model
        pos_hist = [i for i, fb in history if fb == 1] # Ipos
        neg_hist = [i for i, fb in history if fb == 0] # Ineg
        # 计算 alpha_pos, alpha_neg
        Dpos_c = self.item_keywords_pos.get(candidate_item, set())# 候选项目中积极的关键词集合，这里数据中还没有的关键词用set()表示
        Dneg_c = self.item_keywords_neg.get(candidate_item, set())# 候选项目中消极的关键词集合
        alpha_pos = sum(len(Dpos_c & self.item_keywords_pos.get(i, set())) for i in pos_hist)# 候选与历史的课程关键词重合度
        alpha_neg = sum(len(Dneg_c & self.item_keywords_neg.get(i, set())) for i in neg_hist)
        if alpha_pos > alpha_neg:
            return 1
        elif alpha_pos < alpha_neg:
            return 0
        else:
            return np.random.randint(0, 2)# 左闭右开

    def f_sim(self, history, candidate_item):
        # Similarity model

        pos_hist = [i for i, fb in history if fb == 1]
        neg_hist = [i for i, fb in history if fb == 0]
        # 获取候选课程的向量表示
        emb_c = self.get_item_embedding(candidate_item).reshape(1, -1)
        # compute max similarity to pos and neg
        beta_pos = 0.0
        if pos_hist:
            pos_embs = np.vstack([self.get_item_embedding(i) for i in pos_hist])
            beta_pos = float(np.max(cosine_similarity(emb_c, pos_embs)))# 采用余弦相似度
        beta_neg = 0.0
        if neg_hist:
            neg_embs = np.vstack([self.get_item_embedding(i) for i in neg_hist])
            beta_neg = float(np.max(cosine_similarity(emb_c, neg_embs)))
        if beta_pos > beta_neg:
            return 1
        elif beta_pos < beta_neg:
            return 0
        else:
            return np.random.randint(0, 2)

    def f_sta(self, history, candidate_item):
       
        # 在所有交互过的课程中选择喜欢的课程序列作为输入
        item_seq = [i for i, fb in history if fb == 1] #返回积极的课程ID列表
        #如果是新用户，没有看过的或者没有喜欢的则随机推荐
        if not item_seq:
            print("随机推荐")
            return np.random.randint(0, 2)
        states = torch.tensor([item_seq], dtype=torch.long).to(self.device)
        # 计算当前历史交互序列的长度
        len_states = torch.tensor([len(item_seq)], dtype=torch.long).to(self.device)
        # 用SASRec的predict方法预测
        scores = self.statistical_model.predict(states, len_states, topk=len(self.item_pool), eval_mode=True, dim = 0)
        #认为前10个课程是推荐的含有的则给奖励
        if candidate_item in scores[:10].cpu().numpy():
        
            return 1
        else:
            
            return 0

    def predict_feedback(self, history, candidate_item):
        """
        Ensemble decision: majority vote of f_mat, f_sim, f_sta.
        """
        votes = self.f_mat(history, candidate_item) + \
                self.f_sim(history, candidate_item) + \
                self.f_sta(history, candidate_item)
        if votes >= 2:
            reward = 1
        else:
            reward = 0
        return reward
