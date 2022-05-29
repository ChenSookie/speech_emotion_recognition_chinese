#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Sookie
# @File    :
import torch
import torch.nn as nn


class CnnClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super(CnnClassifier, self).__init__()
        self.num_classes = num_classes
        self.conv1d1 = nn.Conv1d(40, 128, 3, padding='same')
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.maxpooling = nn.MaxPool1d(1)
        self.conv1d2 = nn.Conv1d(128, 256, 5, padding='same')
        self.conv1d3 = nn.Conv1d(256, 512, 3, padding='same')
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1d1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.maxpooling(x)
        x = self.conv1d2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.maxpooling(x)
        x = self.conv1d3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.classifier(x)
        score = self.softmax(x)
        return score

