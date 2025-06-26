import os
import numpy as np
import matplotlib.pyplot as plt
import csv
from collections import defaultdict
import random 
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

