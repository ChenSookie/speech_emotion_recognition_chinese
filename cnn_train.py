#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Sookie
# @File    :cnn模型的训练代码

import torch
from sklearn.model_selection import train_test_split
from torch import nn, optim
from classifier.cnnClassifier import CnnClassifier
from utils import evaluate, x_traincnn, y_train, x_testcnn, y_test, cnn_train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CnnClassifier()
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 500
cnn_train(epochs, device, model, optimizer, criterion)


