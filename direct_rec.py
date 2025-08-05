import numpy as np
import csv
import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import pickle
import os
import random
import jieba
import re

#可视化
class Render:
    # 随机推荐
    def random_recommend(user_ids, seed=42, course_min=1, course_max=80, rec_num=10):
        random.seed(int(seed))
        all_pred = defaultdict(list)
        for user_id in user_ids:
            # 随机推荐rec_num个不重复课程编号
            all_pred[user_id] = random.sample(range(course_min, course_max + 1), rec_num)
        # print("random_pred",all_pred)
        return all_pred
    
    # 召回率可视化
    def Recall_analyse(all_pred, all_gt, user_ids):
        hit = 0
        total_pred = 0
        total_gt = 0
        for user_id in user_ids:
            pred_set = set(all_pred[user_id])
            gt_set = set(all_gt[user_id])
            hit += len(pred_set & gt_set)
            total_pred += len(pred_set)
            total_gt += len(gt_set)
        precision = hit / total_pred if total_pred > 0 else 0
        recall = hit / total_gt if total_gt > 0 else 0
        # print("total_gt", total_gt)
        print(f"整体 Precision: {precision:.4f}, Recall: {recall:.4f}")
        return precision, recall

# 检测中英文文本
def detect_language(text):
    """检测文本语言类型"""
    # 统计中文字符数量
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    # 统计英文字符数量
    english_chars = len(re.findall(r'[a-zA-Z]', text))
    
    if chinese_chars > english_chars:
        return 'chinese'
    elif english_chars > 0:
        return 'english'
    else:
        return 'mixed'

def smart_tokenize(text, word_drop_ratio):
    """智能分词函数，支持中英文混合"""
    if not text or text.strip() == "":
        return ""
    
    language = detect_language(text)
    
    if language == 'chinese':
        # 中文使用jieba分词
        words = jieba.lcut(text)
        # 过滤掉空字符串和单个标点符号
        words = [w.strip() for w in words if w.strip() and len(w.strip()) > 0]
    elif language == 'english':
        # 英文使用空格分词
        words = text.split()
    else:
        # 混合文本：先用jieba分词，再对英文部分进行处理
        words = jieba.lcut(text)
        processed_words = []
        for word in words:
            if re.match(r'^[a-zA-Z\s]+$', word):
                # 如果是纯英文，再用空格分割
                processed_words.extend(word.split())
            else:
                processed_words.append(word)
        words = [w.strip() for w in processed_words if w.strip() and len(w.strip()) > 0]
    
    # 应用词汇丢弃
    if word_drop_ratio > 0:
        new_words = []
        for word in words:
            rd = random.random()
            if rd > word_drop_ratio:
                new_words.append(word)
        words = new_words
    
    return ' '.join(words)

def generate_item_embedding(item_text_dic, tokenizer, model, word_drop_ratio=-1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 生成embedding时
    max_item_id = max(item_text_dic.keys())
    order_texts = ["" if k == 0 else item_text_dic.get(k, "") for k in range(max_item_id + 1)]

    embeddings = []
    start, batch_size = 0, 4
    while start < len(order_texts):
        sentences = order_texts[start: start + batch_size]
        
        if word_drop_ratio > 0:
            print(f'Word drop with p={word_drop_ratio}')
            new_sentences = []
            for sent in sentences:
                # 使用智能分词函数
                processed_sent = smart_tokenize(sent, word_drop_ratio)
                new_sentences.append(processed_sent)
            sentences = new_sentences
        
        encoded_sentences = tokenizer(sentences, padding=True, max_length=512,
                                      truncation=True, return_tensors='pt').to(device)
        outputs = model(**encoded_sentences)
        
        # 计算平均池化嵌入
        masked_output = outputs.last_hidden_state * encoded_sentences['attention_mask'].unsqueeze(-1)
        mean_output = masked_output[:,1:,:].sum(dim=1) / encoded_sentences['attention_mask'][:,1:].sum(dim=-1, keepdim=True)
        mean_output = mean_output.detach()
        embeddings.append(mean_output)
        start += batch_size
        
    embeddings = torch.cat(embeddings, dim=0).cpu().numpy()
    print('Embeddings shape: ', embeddings.shape)
    return embeddings

def load_plm(model_name='bert-base-uncased'):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, quantization_config=bnb_config)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model
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
    
    # 归一化
    mat_min, mat_max = min(mat_scores), max(mat_scores)
    sim_min, sim_max = min(sim_scores), max(sim_scores)
    mat_scores_norm = [(s - mat_min) / (mat_max - mat_min) if mat_max > mat_min else 0.0 for s in mat_scores]
    sim_scores_norm = [(s - sim_min) / (sim_max - sim_min) if sim_max > sim_min else 0.0 for s in sim_scores]
    
    # 加权求和
    total_scores = [alpha * m + (1 - alpha) * s for m, s in zip(mat_scores_norm, sim_scores_norm)]
    
    # 排序并返回前k个
    scores = list(zip(candidate_items, total_scores))
    top_k_score = sorted(scores, key=lambda x: x[1], reverse=True)[:k]
    top_k = [item for item, score in top_k_score]
    
    return top_k_score, top_k

def recommend_for_users(user_histories, item_pool, item_keywords_pos, item_keywords_neg, item_embeddings, k=10):
    user_recommendations_scores = {}
    user_recommendations = {}
    for user_id, history in user_histories.items():
        top_k_score, top_k = recommend_top_k(history, item_pool, item_keywords_pos, item_keywords_neg, item_embeddings, k)
        user_recommendations_scores[user_id] = top_k_score
        user_recommendations[user_id] = top_k
    return user_recommendations_scores, user_recommendations

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

user_ids = []
user_profiles = {}
item_pool = []
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
# print(user_recommendations_scores)

# 随机推荐
all_pred_random = Render.random_recommend(user_ids)
# 采用两个逻辑模型处理后的匹配结果
print("逻辑模型推荐结果:")
precision, recall = Render.Recall_analyse(user_recommendations, user_gt_items, user_ids)
# 随机推荐的结果
print("随机推荐结果:")
ran_precision, ran_recall = Render.Recall_analyse(all_pred_random, user_gt_items, user_ids)

# 在文件末尾添加保存和加载函数
def save_recommendation_model(item_keywords_pos, item_keywords_neg, item_embeddings, item_pool, save_path):
    """保存推荐模型的所有组件"""
    # 创建保存目录
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    model_data = {
        'item_keywords_pos': item_keywords_pos,
        'item_keywords_neg': item_keywords_neg, 
        'item_pool': item_pool
    }
    
    # 保存模型参数
    with open(save_path + '_params.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    # 保存嵌入向量
    torch.save(item_embeddings, save_path + '_embeddings.pt')
    
    print(f"模型已保存到: {save_path}")


print("\n" + "="*50)
print("保存推荐模型...")
save_recommendation_model(item_keywords_pos, item_keywords_neg, item_embeddings, item_pool, 
                         'DRL-code-pytorch-main/Course_DQN/saved_models/direct_rec_model')

