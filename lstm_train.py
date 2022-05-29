# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Sookie
# @File    :LSTM模型的训练代码
import torch
from torch import optim, nn
from classifier.LstmClassifier import LstmClassifier
from utils import lstm_train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LstmClassifier()
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 500
lstm_train(epochs, device, model, criterion, optimizer)
