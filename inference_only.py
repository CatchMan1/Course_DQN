import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# 复制必要的函数
def f_mat(history, candidate_item, item_keywords_pos, item_keywords_neg):
    pos_hist = [i for i, fb in history if fb == 1]
    neg_hist = [i for i, fb in history if fb == 0]
    Dpos_c = item_keywords_pos.get(candidate_item, set())
    Dneg_c = item_keywords_neg.get(candidate_item, set())
    alpha_pos = sum(len(Dpos_c & item_keywords_pos.get(i, set())) for i in pos_hist)
    alpha_neg = sum(len(Dneg_c & item_keywords_neg.get(i, set())) for i in neg_hist)
    return alpha_pos - alpha_neg

def f_sim(history, candidate_item, item_embeddings):
    pos_hist = [i for i, fb in history if fb == 1]
    neg_hist = [i for i, fb in history if fb == 0]
    emb_c = item_embeddings[candidate_item].cpu().numpy().reshape(1, -1)
    beta_pos = 0.0
    if pos_hist:
        pos_embs = np.vstack([item_embeddings[i].cpu().numpy() for i in pos_hist])
        beta_pos = float(np.max(cosine_similarity(emb_c, pos_embs)))
    beta_neg = 0.0
    if neg_hist:
        neg_embs = np.vstack([item_embeddings[i].cpu().numpy() for i in neg_hist])
        beta_neg = float(np.max(cosine_similarity(emb_c, neg_embs)))
    return beta_pos - beta_neg

def recommend_top_k(user_history, item_pool, item_keywords_pos, item_keywords_neg, item_embeddings, k=10, alpha=0.5):
    # 提取用户已交互过的项目ID
    interacted_items = set([item_id for item_id, _ in user_history])
    
    # 过滤掉已交互的项目
    candidate_items = [item for item in item_pool if item not in interacted_items]
    
    if len(candidate_items) == 0:
        print("警告：没有可推荐的新项目")
        return [], []
    
    mat_scores = []
    sim_scores = []
    
    # 只对未交互的项目计算分数
    for item in candidate_items:
        mat_score = f_mat(user_history, item, item_keywords_pos, item_keywords_neg)
        sim_score = f_sim(user_history, item, item_embeddings)
        mat_scores.append(mat_score)
        sim_scores.append(sim_score)
    
    # 归一化和排序逻辑保持不变
    mat_min, mat_max = min(mat_scores), max(mat_scores)
    sim_min, sim_max = min(sim_scores), max(sim_scores)
    mat_scores_norm = [(s - mat_min) / (mat_max - mat_min) if mat_max > mat_min else 0.0 for s in mat_scores]
    sim_scores_norm = [(s - sim_min) / (sim_max - sim_min) if sim_max > sim_min else 0.0 for s in sim_scores]
    
    total_scores = [alpha * m + (1 - alpha) * s for m, s in zip(mat_scores_norm, sim_scores_norm)]
    scores = list(zip(candidate_items, total_scores))
    top_k_score = sorted(scores, key=lambda x: x[1], reverse=True)[:k]
    top_k = [item for item, score in top_k_score]
    
    return top_k_score, top_k

# 模型加载
def load_and_recommend(model_path, user_history, k=10):
    # 加载模型参数
    with open(model_path + '_params.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    # 加载嵌入向量
    item_embeddings = torch.load(model_path + '_embeddings.pt')
    
    # 进行推荐
    top_k_score, top_k = recommend_top_k(
        user_history, 
        model_data['item_pool'], 
        model_data['item_keywords_pos'], 
        model_data['item_keywords_neg'], 
        item_embeddings, 
        k
    )
    
    return top_k_score, top_k

def API_rec(user_item, model_path, k=10):
    
    recommendations = {}
    
    try:
        # 加载模型参数
        with open(model_path + '_params.pkl', 'rb') as f:
            model_data = pickle.load(f)
        
        # 加载嵌入向量
        item_embeddings = torch.load(model_path + '_embeddings.pt')
        
        for user_id, user_history in user_item.items():
            try:
                if user_history:  # 确保用户有历史记录
                    # 调用推荐函数
                    top_k_score, top_k = recommend_top_k(
                        user_history, 
                        model_data['item_pool'], 
                        model_data['item_keywords_pos'], 
                        model_data['item_keywords_neg'], 
                        item_embeddings, 
                        k
                    )
                    recommendations[user_id] = top_k  # 只返回课程ID列表
                else:
                    print(f"用户 {user_id} 没有历史记录，跳过推荐")
                    recommendations[user_id] = []
                    
            except Exception as e:
                print(f"为用户 {user_id} 生成推荐失败: {str(e)}")
                recommendations[user_id] = []
                
    except Exception as e:
        print(f"加载模型失败: {str(e)}")
        # 如果模型加载失败，为所有用户返回空列表
        recommendations = {user_id: [] for user_id in user_item.keys()}
    return recommendations

if __name__ == "__main__":
    # 示例使用
    model_path = 'DRL-code-pytorch-main/Course_DQN/saved_models/direct_rec_model'

    user_item = {"a":[(1, 1), (3, 1), (5, 0), (7, 1)],
                 "b":[(1, 0), (16, 1), (6, 0), (9, 1)],
                 "c":[(1, 0), (2, 1), (3, 0), (4, 1)],
                 "d":[(11, 1), (27, 0), (3, 1), (4, 0)],
                 "e":[(1, 0), (28, 1), (3, 0), (4, 1)],
                 "f":[(1, 1), (2, 0), (39, 1), (43, 0)],}
    
    # 调用推荐API
    recommendations = API_rec(user_item, model_path, k=10)
    
    # 输出结果
    print("推荐结果:")
    print(recommendations)