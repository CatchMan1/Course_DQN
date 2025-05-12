import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class UserSimulator:
  
    def __init__(self,user_id,item_pool,item_keywords_pos,item_keywords_neg,
                 item_embeddings,user_profile_emb,init_history,statistical_model=None):
        self.user_id = user_id
        self.item_pool = item_pool
        self.item_keywords_pos = item_keywords_pos
        self.item_keywords_neg = item_keywords_neg
        self.item_embeddings = item_embeddings
        self.user_profile_emb = user_profile_emb
        self.history = list(init_history) if init_history is not None else []# 保留副本，减少风险
        self.stat_model = statistical_model
        self.item_emb_dim = next(iter(item_embeddings.values())).shape[0] # 取第一个item的维度
        # Precompute embedding matrix if needed
        self.item_emb_matrix = np.vstack([item_embeddings[i] for i in item_pool])
        self.item_index = {i: idx for idx, i in enumerate(item_pool)}

    def get_user_embedding(self):
        
        return self.user_profile_emb

    def get_item_embedding(self, item_id):
        
        return self.item_embeddings[item_id]

    def f_mat(self, history, candidate_item):
        # Keyword matching model

        pos_hist = [i for i, fb in history if fb == 1] # Ipos
        neg_hist = [i for i, fb in history if fb == 0] # Ineg
        # If no same-category items, use all history (omitted here)
        # Compute alpha_pos, alpha_neg
        Dpos_c = self.item_keywords_pos.get(candidate_item, set())# 候选项目中积极的关键词集合
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
       
        if self.stat_model is not None:
            return self.stat_model.predict(history, candidate_item)
        else:
            # 先用（0，2）的随机数代替统计模型
            #print("没调用成功")
            return np.random.randint(0, 2)

    def predict_feedback(self, history, candidate_item):
        """
        Ensemble decision: majority vote of f_mat, f_sim, f_sta.
        """
        votes = self.f_mat(history, candidate_item) + \
                self.f_sim(history, candidate_item) + \
                self.f_sta(history, candidate_item)
        return 1 if votes >= 2 else 0
