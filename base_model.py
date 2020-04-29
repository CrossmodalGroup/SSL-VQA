"""
This code is modified from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier, PaperClassifier
from fc import FCNet, GTH
from attention import Att_0, Att_1, Att_2, Att_3, Att_P, Att_PD, Att_3S
import torch
import random


class Model(nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()
        num_hid = opt.num_hid
        activation = opt.activation
        dropG = opt.dropG
        dropW = opt.dropW
        dropout = opt.dropout
        dropL = opt.dropL
        norm = opt.norm
        dropC = opt.dropC
        self.opt = opt

        self.w_emb = WordEmbedding(opt.ntokens, emb_dim=300, dropout=dropW)
        self.w_emb.init_embedding(opt.dataroot + 'glove6b_init_300d.npy')
        self.q_emb = QuestionEmbedding(in_dim=300, num_hid=num_hid, nlayers=1,
                                       bidirect=False, dropout=dropG, rnn_type='GRU')

        self.q_net = FCNet([self.q_emb.num_hid, num_hid], dropout=dropL, norm=norm, act=activation)
        self.gv_net = FCNet([opt.v_dim, num_hid], dropout=dropL, norm=norm, act=activation)

        self.gv_att_1 = Att_3(v_dim=opt.v_dim, q_dim=self.q_emb.num_hid, num_hid=num_hid, dropout=dropout, norm=norm,
                              act=activation)
        self.gv_att_2 = Att_3(v_dim=opt.v_dim, q_dim=self.q_emb.num_hid, num_hid=num_hid, dropout=dropout, norm=norm,
                              act=activation)
        self.classifier = SimpleClassifier(in_dim=num_hid, hid_dim=2 * num_hid, out_dim=opt.ans_dim,
                                           dropout=dropC, norm=norm, act=activation)

        self.normal = nn.BatchNorm1d(num_hid,affine=False)

    def forward(self, q, gv_pos, self_sup=True):

        """Forward
        q: [batch_size, seq_length]
        gv_pos: [batch, K, v_dim]
        self_sup: use negative images or not
        return: logits, not probs
        """

        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb)  # run GRU on word embeddings [batch, q_dim]
        q_repr = self.q_net(q_emb)
        batch_size = q.size(0)

        logits_pos, att_gv_pos = self.compute_predict(q_repr, q_emb, gv_pos)

        if self_sup:
            # construct an irrelevant Q-I pair for each instance
            index = random.sample(range(0, batch_size), batch_size)
            gv_neg = gv_pos[index]
            logits_neg, att_gv_neg = \
                self.compute_predict(q_repr, q_emb, gv_neg)
            return logits_pos, logits_neg, att_gv_pos, att_gv_neg
        else:
            return logits_pos, att_gv_pos

    def compute_predict(self, q_repr, q_emb, v):

        att_1 = self.gv_att_1(v, q_emb)  # [batch, 1, v_dim]
        att_2 = self.gv_att_2(v, q_emb)  # [batch, 1, v_dim]
        att_gv = att_1 + att_2

        gv_embs = (att_gv * v)  # [batch, v_dim]
        gv_emb = gv_embs.sum(1)
        gv_repr = self.gv_net(gv_emb)

        joint_repr = q_repr * gv_repr

        joint_repr_normal = self.normal(joint_repr)
        logits = self.classifier(joint_repr_normal)

        return logits, att_gv

