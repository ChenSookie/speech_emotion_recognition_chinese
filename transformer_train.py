#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Sookie
# @File    :transformer模型的训练代码
import os

import numpy as np
import torch
from torch import nn

from classifier.transformerClassifier import Classifier
from utils import transformer_train, load_checkpoint, validate

filename = 'lib/features_labels.npy'
with open(filename, 'rb') as f:
    X_train = np.load(f)
    X_valid = np.load(f)
    X_test = np.load(f)
    Y_train = np.load(f)
    Y_valid = np.load(f)
    Y_test = np.load(f)

minibatch = 42
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'{device} selected')
model = Classifier(6).to(device)
print('Number of trainable params: ', sum(p.numel() for p in model.parameters()))
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-3, momentum=0.8)

num_epochs = 350
transformer_train(minibatch, device, model, optimizer, num_epochs)

load_folder = 'result/'
epoch = '123'
model_name = f'transformer_model-{epoch}.pkl'
load_path = os.path.join(load_folder, model_name)

criterion = nn.CrossEntropyLoss()
load_checkpoint(optimizer, model, load_path)
print(f'Loaded model from {load_path}')

X_test_tensor = torch.tensor(X_test, device=device).float()
y_test_tensor = torch.tensor(Y_test, dtype=torch.long, device=device)
test_loss, test_acc, predicted_emotions = validate(X_test_tensor, y_test_tensor, model, criterion)
print(f'Test accuracy is {test_acc:.2f}%')
