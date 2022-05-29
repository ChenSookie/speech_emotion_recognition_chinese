#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Sookie
# @File    :
import torch.nn as nn
import torch.nn.functional as F


class LstmClassifier(nn.Module):
    def __init__(self):
        super(LstmClassifier, self).__init__()
        self.n_layers = 2
        self.input_dim = 40
        self.hidden_dim = 256
        self.output_dim = 6
        self.bidirectional = False
        self.dropout = 0.2 if self.n_layers > 1 else 0

        self.rnn = nn.LSTM(self.input_dim, self.hidden_dim, bias=True,
                           num_layers=self.n_layers, dropout=self.dropout,
                           bidirectional=self.bidirectional)
        self.out = nn.Linear(self.hidden_dim, self.output_dim)
        self.softmax = F.softmax

    def forward(self, input_seq):
        rnn_output, (hidden, _) = self.rnn(input_seq)
        if self.bidirectional:
            rnn_output = rnn_output[:, :, :self.hidden_dim] + \
                         rnn_output[:, :, self.hidden_dim:]
        class_scores = F.softmax(self.out(rnn_output[0]), dim=1)
        return class_scores

