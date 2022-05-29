#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 
# @Author  : Sookie
# @File    :  evaluate函数, train函数, 特征缩放函数
import gc
import pickle
import joblib
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn


X = joblib.load('lib/X.joblib')
y = joblib.load('lib/y.joblib')
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=69)
print(X.shape, y.shape)
x_traincnn = np.expand_dims(x_train, axis=2)
x_testcnn = np.expand_dims(x_test, axis=2)
print(x_traincnn.shape, x_testcnn.shape)
sample_rate = 22050


# 特征缩放
def X_Scaler(S):
    scaler = StandardScaler()
    N, C, H, W = S.shape
    S = np.reshape(S, (N, -1))
    S = scaler.fit_transform(S)
    S = np.reshape(S, (N, C, H, W))
    return S


# 评估
def evaluate(targets, predictions):
    performance = {
        'acc': accuracy_score(targets, predictions),
        'f1': f1_score(targets, predictions, average='macro'),
        'precision': precision_score(targets, predictions, average='macro'),
        'recall': recall_score(targets, predictions, average='macro')}
    return performance


# lstm模型训练
def lstm_train(epochs, device, model, criterion, optimizer):
    best_acc = 0
    train_batches = [[torch.FloatTensor(x_train), torch.LongTensor(y_train)]]
    test_pairs = [torch.FloatTensor(x_test), torch.LongTensor(y_test)]
    for epoch in range(epochs):
        train_losses = []
        for batch in train_batches:
            inputs = batch[0].unsqueeze(0)
            targets = batch[1]
            inputs = inputs.to(device)
            targets = targets.to(device)
            model.zero_grad()
            optimizer.zero_grad()
            predictions = model(inputs)
            predictions = predictions.to(device)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        with torch.no_grad():
            inputs = test_pairs[0].unsqueeze(0)
            targets = test_pairs[1]
            inputs = inputs.to(device)
            targets = targets.to(device)
            predictions = torch.argmax(model(inputs), dim=1)  # take argmax to get class id
            predictions = predictions.to(device)
            # 模型的评估
            targets = np.array(targets.cpu())
            predictions = np.array(predictions.cpu())
            performance = evaluate(targets, predictions)
            if performance['acc'] > best_acc:
                best_acc = performance['acc']
                print(performance)
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, 'result/{}-best_model.pth'.format('basic_lstm'))

                with open('result/{}-best_performance.pkl'.format('basic_lstm'), 'wb') as f:
                    pickle.dump(performance, f)


# cnn模型训练
def cnn_train(epochs, device, model, optimizer, criterion):
    best_acc = 0
    for epoch in range(epochs):
        losses = []
        inputs = torch.FloatTensor(x_traincnn)
        targets = torch.LongTensor(y_train)
        inputs = inputs.to(device)
        targets = targets.to(device)
        model.zero_grad()
        optimizer.zero_grad()
        predictions = model(inputs)
        predictions = predictions.to(device)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        with torch.no_grad():
            inputs = torch.FloatTensor(x_testcnn)
            targets = torch.LongTensor(y_test)
            inputs = inputs.to(device)
            targets = targets.to(device)
            predictions = torch.argmax(model(inputs), dim=1)
            predictions = predictions.to(device)
            # 模型的评估
            targets = np.array(targets.cpu())
            predictions = np.array(predictions.cpu())
            performance = evaluate(targets, predictions)
            if performance['acc'] > best_acc:
                best_acc = performance['acc']
                print(performance)
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, 'result/{}-best_cnn_model.pth'.format('cnn'))

                with open('result/{}-best_cnn_performance.pkl'.format('cnn'), 'wb') as f:
                    pickle.dump(performance, f)


# transformer模型训练
def transformer_train(minibatch, device, model, optimizer, num_epochs):
    filename = 'lib/features_labels.npy'
    with open(filename, 'rb') as f:
        X_train = np.load(f)
        X_valid = np.load(f)
        X_test = np.load(f)
        Y_train = np.load(f)
        Y_valid = np.load(f)
        Y_test = np.load(f)

    def train_step(X, Y):
        output_logits, output_softmax = model(X)
        predictions = torch.argmax(output_softmax, dim=1)
        accuracy = torch.sum(Y == predictions) / float(len(Y))
        loss = criterion(output_logits, Y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item(), accuracy * 100

    def save_checkpoint(optimizer, model, epoch, filename):
        checkpoint_dict = {
            'optimizer': optimizer.state_dict(),
            'model': model.state_dict(),
            'epoch': epoch
        }
        torch.save(checkpoint_dict, filename)

    criterion = nn.CrossEntropyLoss()
    train_losses = []
    valid_losses = []
    train_size = X_train.shape[0]
    for epoch in range(num_epochs):
        gc.collect()
        torch.cuda.empty_cache()
        model.train()
        train_indices = np.random.permutation(train_size)
        X_train = X_train[train_indices, :, :, :]
        Y_train = Y_train[train_indices]
        epoch_acc = 0
        epoch_loss = 0
        num_iterations = int(train_size / minibatch)
        for i in range(num_iterations):  # 分批次训练
            batch_start = i * minibatch
            batch_end = min(batch_start + minibatch, train_size)
            actual_batch_size = batch_end - batch_start
            X = X_train[batch_start:batch_end, :, :, :]
            Y = Y_train[batch_start:batch_end]
            X_tensor = torch.tensor(X, device=device).float()
            Y_tensor = torch.tensor(Y, dtype=torch.long, device=device)
            loss, acc = train_step(X_tensor, Y_tensor)
            epoch_acc += acc * actual_batch_size / train_size
            epoch_loss += loss * actual_batch_size / train_size
            print('\r' + f'Epoch {epoch}: iteration {i}/{num_iterations}', end='')
        X_valid_tensor = torch.tensor(X_valid, device=device).float()
        Y_valid_tensor = torch.tensor(Y_valid, dtype=torch.long, device=device)
        with torch.no_grad():
            valid_loss, valid_acc, _ = validate(X_valid_tensor, Y_valid_tensor, model, criterion)
        train_losses.append(epoch_loss)
        valid_losses.append(valid_loss)
        checkpoint_filename = 'result/transformer_model-{:03d}.pkl'.format(epoch)
        save_checkpoint(optimizer, model, epoch, checkpoint_filename)
        print(
            f'\nEpoch {epoch} --- loss:{epoch_loss:.3f}, Epoch accuracy:{epoch_acc:.2f}%, '
            f'Validation loss:{valid_loss:.3f}, Validation accuracy:{valid_acc:.2f}%')
        gc.collect()
        torch.cuda.empty_cache()
    plt.title('Loss Curve for  Model')
    plt.ylabel('Loss', fontsize=16)
    plt.xlabel('Epoch', fontsize=16)
    plt.plot(train_losses[:], 'b')
    plt.plot(valid_losses[:], 'r')
    plt.legend(['Training loss', 'Validation loss'])
    plt.show()


def load_checkpoint(optimizer, model, filename):
    checkpoint_dict = torch.load(filename)
    epoch = checkpoint_dict['epoch']
    model.load_state_dict(checkpoint_dict['model'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
    return epoch


# transformer模型的验证
def validate(X, Y, model, criterion):
    with torch.no_grad():
        model.eval()
        output_logits, output_softmax = model(X)  # 因为LSTM和cnn模型的输出只有softmax不能用这个函数
        predictions = torch.argmax(output_softmax, dim=1)
        accuracy = torch.sum(Y == predictions) / float(len(Y))
        loss = criterion(output_logits, Y)
    return loss.item(), accuracy * 100, predictions







