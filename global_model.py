import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from Aggregator import RGCNAggregator_global
from utils import *
import time


class TransformerHidden(nn.Module):
    def __init__(self, d_model, target_size, nhead):
        super(TransformerHidden, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.target_size = target_size
        self.linear = nn.Linear(d_model, target_size)
<<<<<<< HEAD
        # self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        # self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        self.transformer = nn.Transformer(d_model=d_model,nhead=nhead, num_encoder_layers=12)

    def forward(self, src):
        target_list = []
        all_tensor = self.transformer(src, src)
=======
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)

    def forward(self, src):
        target_list = []
        all_tensor = self.transformer_encoder(src)
>>>>>>> 71f917b1fef5524c476746464d43b14f35f21cdb

        for i in range(src.shape[0]):
            target_vector = all_tensor[i][-1]
            target_list.append(target_vector.unsqueeze(0))
        final_result = torch.cat(target_list, dim=0)
        return self.linear(final_result).unsqueeze(0)


class RENet_global(nn.Module):
    def __init__(self, entity_tensor, in_dim, h_dim, num_rels, dropout=0, model=0, seq_len=10, num_k=10, maxpool=1):
        super(RENet_global, self).__init__()
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.num_rels = num_rels
        self.model = model
        self.seq_len = seq_len
        self.num_k = num_k
        self.linear_entity = nn.Linear(768, h_dim)
        self.ent_embeds = self.linear_entity(entity_tensor)
        nn.init.xavier_uniform_(self.ent_embeds,
                                gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)
        self.transformer_hidden = TransformerHidden(d_model=h_dim, nhead=10, target_size=h_dim)
        # 考虑是否换成LSTM或者Transformer
        self.encoder_global = nn.GRU(h_dim, h_dim, batch_first=True)
        encoder_layer = nn.TransformerEncoderLayer(d_model=h_dim, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

        self.aggregator = RGCNAggregator_global(h_dim, dropout, in_dim, num_rels, 100, model, seq_len, maxpool)

        self.linear_s = nn.Linear(h_dim, in_dim)
        self.linear_o = nn.Linear(h_dim, in_dim)
        self.global_emb = None

    def forward(self, t_list, true_prob_s, true_prob_o, graph_dict, subject=True):
        if subject:
            reverse = False
            linear = self.linear_s
            true_prob = true_prob_o
        else:
            reverse = True
            linear = self.linear_o
            true_prob = true_prob_s

        sorted_t, idx = t_list.sort(0, descending=True)

        packed_input = self.aggregator(sorted_t, self.ent_embeds, graph_dict, reverse=reverse)

        #        tt, s_q = self.encoder_global(packed_input)
        #        transformer_hidden = TransformerHidden(d_model=self.h_dim, nhead=10, target_size=self.h_dim)
        #        print("packed_input_shape")
        #        print(packed_input.shape)
        s_q = self.transformer_hidden(packed_input)
        s_q = s_q.squeeze()
        s_q = torch.cat((s_q, torch.zeros(len(t_list) - len(s_q), self.h_dim).cuda()), dim=0)
        pred = linear(s_q)
        loss = soft_cross_entropy(pred, true_prob[idx])

        return loss

    def get_global_emb(self, t_list, graph_dict):
        global_emb = dict()
        times = list(graph_dict.keys())
        time_unit = times[1] - times[0]

        prev_t = 0
        for t in t_list:
            if t == 0:
                continue

            emb, _, _ = self.predict(t, graph_dict)
            global_emb[prev_t] = emb.detach_()
            prev_t = t

        global_emb[t_list[-1]], _, _ = self.predict(t_list[-1] + int(time_unit), graph_dict)
        global_emb[t_list[-1]].detach_()
        return global_emb

    """
    Prediction function in testing
    """

    def predict(self, t, graph_dict, subject=True):  # Predict s at time t, so <= t-1 graphs are used.
        if subject:
            linear = self.linear_s
            reverse = False
        else:
            linear = self.linear_o
            reverse = True
        rnn_inp = self.aggregator.predict(t, self.ent_embeds, graph_dict, reverse=reverse)
        tt, s_q = self.encoder_global(rnn_inp.view(1, -1, self.h_dim))
        sub = linear(s_q)
        prob_sub = torch.softmax(sub.view(-1), dim=0)
        return s_q, sub, prob_sub

    def update_global_emb(self, t, graph_dict):
        pass
