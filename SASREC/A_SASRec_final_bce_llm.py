import numpy as np
import pandas as pd
import argparse
import torch
from torch import nn
import torch.nn.functional as F
import os
import logging
import time as Time
from collections import Counter
from SASREC.SASRecModules_ori import *
# from SASRecModules_ori import *
# 提取每个序列最后一个有效位置的隐藏状态，作为序列整体特征的表示
def extract_axis_1(data, indices):
    res = []
    for i in range(data.shape[0]):
        res.append(data[i, indices[i], :])
    res = torch.stack(res, dim=0).unsqueeze(1)
    return res


class SASRec(nn.Module):
    def __init__(self, hidden_size, item_num, state_size, dropout, device, num_heads=1, pre_embeddings=None, proj_dim=8):
        super().__init__()
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.proj_dim = proj_dim
        self.item_num = int(item_num)
        self.dropout = nn.Dropout(dropout)
        self.device = device

        self.item_embeddings = torch.nn.Embedding.from_pretrained(pre_embeddings, freeze=False)
        self.act = nn.ReLU()

        self.positional_embeddings = nn.Embedding(
            num_embeddings=state_size,
            embedding_dim=hidden_size
        )
        self.emb_dropout = nn.Dropout(dropout)
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.ln_2 = nn.LayerNorm(hidden_size)
        self.ln_3 = nn.LayerNorm(hidden_size)
        self.mh_attn = MultiHeadAttention(hidden_size, hidden_size, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size, hidden_size, dropout)

        # 多层降维：768  256  64  16（proj_dim）
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 16),
            nn.ReLU(),
            nn.Linear(16, proj_dim)
        )

        self.s_fc = nn.Linear(proj_dim, item_num)

    def forward(self, states, len_states):
        states = states.to(self.device)
        inputs_emb = self.item_embeddings(states)
        seq_length = states.size(1)
        seq = self.emb_dropout(inputs_emb)
        mask = torch.ne(states, self.item_num).float().unsqueeze(-1).to(self.device)
        seq *= mask

        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.mh_attn(seq_normalized, seq)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)

        state_hidden = extract_axis_1(ff_out, len_states - 1)     # [B, hidden_size]
        state_hidden = self.projection(state_hidden)              # [B, proj_dim=16]
        supervised_output = self.s_fc(state_hidden).squeeze()
        return supervised_output

    def forward_eval(self, states, len_states):
        states = states.to(self.device)
        inputs_emb = self.item_embeddings(states)
        seq_length = states.size(1)
        positions = torch.arange(seq_length, device=self.positional_embeddings.weight.device).unsqueeze(0)
        inputs_emb += self.positional_embeddings(positions)

        seq = self.emb_dropout(inputs_emb)
        mask = torch.ne(states, self.item_num).float().unsqueeze(-1).to(self.device)
        seq *= mask

        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.mh_attn(seq_normalized, seq)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)

        state_hidden = extract_axis_1(ff_out, len_states - 1)
        state_hidden = self.projection(state_hidden)
        supervised_output = self.s_fc(state_hidden).squeeze()
        return supervised_output

    def cacul_h(self, states, len_states):
        inputs_emb = self.item_embeddings(states)
        inputs_emb += self.positional_embeddings(torch.arange(self.state_size).to(self.device))
        seq = self.emb_dropout(inputs_emb)
        mask = torch.ne(states, self.item_num).float().unsqueeze(-1).to(self.device)
        seq *= mask

        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.mh_attn(seq_normalized, seq)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)

        state_hidden = extract_axis_1(ff_out, len_states - 1)
        state_hidden = self.projection(state_hidden)  # 降维后输出
        return state_hidden

    def cacu_x(self, x):
        x = self.item_embeddings(x)
        return x

    def predict(self, states, len_states, topk=10, eval_mode=True, dim=1):
        self.eval()
        with torch.no_grad():
            if eval_mode:
                output = self.forward_eval(states, len_states)
            else:
                output = self.forward(states, len_states)
            _, predicted_indices = torch.topk(output, k=topk, dim=dim)
        return predicted_indices
