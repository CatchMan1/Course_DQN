import torch
import numpy as np
class SASRecWrapper:
    def __init__(self, sasrec_model, item2id, maxlen, device='cpu'):
        # 调用SAS模型的评估模式
        self.model = sasrec_model.eval()
        self.item2id = item2id  # dict[str, int]，将课程名映射到 ID
        self.maxlen = maxlen    # 序列最大长度（与训练时一致）
        self.device = device

    def predict(self, history, candidate_item):
        # history: List[Tuple[item_id:str, feedback:int]]
        # candidate_item: str

        # 提取 item ID 列表，最多 maxlen 个（按时间顺序）
        item_seq = [self.item2id[i] for i, _ in history][-self.maxlen:]

        # Padding（前面补 0）
        pad_len = self.maxlen - len(item_seq)
        item_seq = [0] * pad_len + item_seq  # 0 为 padding_idx

        # 构造输入
        user_ids = [0]  # 单用户
        log_seqs = np.array([item_seq])
        candidate_id = self.item2id[candidate_item]
        item_indices = [candidate_id]

        # 得到打分
        with torch.no_grad():
            scores = self.model.predict(user_ids, log_seqs, item_indices)  # shape: (1, 1)
            prob = torch.sigmoid(scores[0, 0]).item()
        
        # 二值化输出
        return 1 if prob > 0.5 else 0
