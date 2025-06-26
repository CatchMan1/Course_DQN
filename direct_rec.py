import numpy as np
import csv
import torch
from sklearn.metrics.pairwise import cosine_similarity
from SASREC.Rec import load_plm, generate_item_embedding
from render import Render
# 用户简档尚未加入进去计算
# 这里是计算候选项目与用户历史项目之间的相似度（关键词匹配，项目未嵌入，采用文本分词匹配）
def f_mat(history, candidate_item, item_keywords_pos, item_keywords_neg):
    # Keyword matching model
    pos_hist = [i for i, fb in history if fb == 1] # Ipos
    neg_hist = [i for i, fb in history if fb == 0] # Ineg
    Dpos_c = item_keywords_pos.get(candidate_item, set())
    Dneg_c = item_keywords_neg.get(candidate_item, set())
    alpha_pos = sum(len(Dpos_c & item_keywords_pos.get(i, set())) for i in pos_hist)
    alpha_neg = sum(len(Dneg_c & item_keywords_neg.get(i, set())) for i in neg_hist)
    return alpha_pos - alpha_neg  # 返回差值作为分数

# 这里是计算候选项目与用户历史项目之间的相似度（项目嵌入embedding）
def f_sim(history, candidate_item, item_embeddings):
    # Similarity model
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
    return beta_pos - beta_neg  # 返回差值作为分数

def recommend_top_k(user_history, item_pool, item_keywords_pos, item_keywords_neg, item_embeddings, k=10, alpha=0.5):
    mat_scores = []
    sim_scores = []
    for item in item_pool:
        mat_score = f_mat(user_history, item, item_keywords_pos, item_keywords_neg)
        sim_score = f_sim(user_history, item, item_embeddings)
        mat_scores.append(mat_score)
        sim_scores.append(sim_score)
    # 归一化
    mat_min, mat_max = min(mat_scores), max(mat_scores)
    sim_min, sim_max = min(sim_scores), max(sim_scores)
    mat_scores_norm = [(s - mat_min) / (mat_max - mat_min) if mat_max > mat_min else 0.0 for s in mat_scores]
    sim_scores_norm = [(s - sim_min) / (sim_max - sim_min) if sim_max > sim_min else 0.0 for s in sim_scores]
    # 加权求和
    total_scores = [alpha * m + (1 - alpha) * s for m, s in zip(mat_scores_norm, sim_scores_norm)]
    # 排序并返回前k
    scores = list(zip(item_pool, total_scores))
    top_k_socre = sorted(scores, key=lambda x: x[1], reverse=True)[:k]
    top_k = [item for item, score in top_k_socre]
    return top_k_socre, top_k  # 返回分数和推荐列表

def recommend_for_users(user_histories, item_pool, item_keywords_pos, item_keywords_neg, item_embeddings, k=10):
    user_recommendations_scores = {}
    user_recommendations = {}
    for user_id, history in user_histories.items():
        top_k_score, top_k = recommend_top_k(history, item_pool, item_keywords_pos, item_keywords_neg, item_embeddings, k)
        user_recommendations_scores[user_id] = top_k_score
        user_recommendations[user_id] = top_k
    return user_recommendations_scores, user_recommendations


user_ids = []
user_profiles = {}
item_pool = []
# 所有用户以及用户简档
# ./SASREC/ai_user_summary_output_fixed.csv'
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
            item_pool.append(item_id)


item_keywords_pos = {}
item_keywords_neg = {}

with open('DRL-code-pytorch-main/Course_DQN/class_index.csv','r',encoding='utf-8') as f2:
    reader = csv.reader(f2)
    next(reader)
    for row in reader:
        item_id = int(row[0])
        item_keywords_pos[item_id] = set(row[3].split())
        item_keywords_neg[item_id] = set(row[4].split())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

plm_tokenizer, plm_model = load_plm('bert-base-uncased')
plm_model = plm_model.to(device)


item_text_dic = {}
item_text_dic = {
    item_id: " ".join(item_keywords_pos.get(item_id, set()) | item_keywords_neg.get(item_id, set()))
    for item_id in item_pool
}
item_text_dic[0] = ""  # 添加 padding 项目，用空字符串（作为嵌入的index=0的位置），保证课程编号从1开始正确匹配
# 批量生成item_embeddings 维度为（81，768）的tensor, 内容包含了课程的积极和消极关键词文本的嵌入
item_embeddings = generate_item_embedding(item_text_dic, plm_tokenizer, plm_model, word_drop_ratio=-1)
item_embeddings = torch.tensor(item_embeddings, dtype=torch.float32).to(device)

# 训练数据
user_histories = {}
# 测试数据
user_gt_items = {}
for user_id in user_ids:
    history_item = []
    with open(f'DRL-code-pytorch-main/Course_DQN/student_info/{user_id}.csv', 'r', encoding='utf-8') as f1:
        reader = csv.reader(f1)
        next(reader)  # 跳过表头
        for row in reader:
            item_id = int(row[0])
            history_item.append((item_id, int(row[4])))
        if len(history_item) > 4:
                train_history_item = history_item[:-4]
                test_gt_items = [item for item, fb in history_item[-4:] if fb == 1]
        else:
            train_history_item = []
            test_gt_items = [item for item, fb in history_item if fb == 1]

        user_histories[user_id] = train_history_item
        user_gt_items[user_id] = test_gt_items

user_recommendations_scores, user_recommendations = recommend_for_users(user_histories, item_pool, item_keywords_pos, item_keywords_neg, item_embeddings, k=10)
# 打印出每个用户的推荐结果和倒序分数
print(user_recommendations_scores)

# 随机推荐
all_pred_random = Render.random_recommend(user_ids)
# 采用两个逻辑模型处理后的匹配结果
precision, recall = Render.Recall_analyse(user_recommendations, user_gt_items, user_ids)
# 随机推荐的结果
ran_precision, ran_recall = Render.Recall_analyse(all_pred_random, user_gt_items, user_ids)