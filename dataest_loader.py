#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Sookie
# @File    :transformer模型用到的特征向量数据格式
import fnmatch
import os
import librosa
import numpy as np

from utils import X_Scaler

emo_class = 6
labsIndName = []
sample_rate = 22050
path = "CASIA_database"
actors = os.listdir(path)
actors = fnmatch.filter(actors, "actor_*")
waveforms = []
emotions = []
emotions_dic = {'angry': 0, 'fear': 1, 'happy': 2, 'neutral': 3, 'sad': 4, 'surprise': 5}
dirs = fnmatch.filter(actors, "actor_*")


# 加载音频数据
for actor in actors:
    dirs = os.listdir(path + '/' + actor)
    for i in dirs:
        print("开始加载：", i)
        labsIndName.append(i)
        print(labsIndName)
        wav_path = path + "/" + actor + "/" + i
        print(wav_path)
        files = os.listdir(wav_path)
        for j in fnmatch.filter(files, "*.wav"):  # 这里跟cnn和LSTM模型的处理不一样
            file = wav_path + "/" + j
            Waveform, _ = librosa.load(file, duration=3, offset=0.5, sr=sample_rate)
            waveform_h = np.zeros((int(sample_rate * 3)))
            waveform_h[:len(Waveform)] = Waveform
            emotion = emotions_dic[i]
            waveforms.append(waveform_h)
            emotions.append(emotion)
# 划分数据集
train_set, valid_set, test_set = [], [], []
X_train, X_valid, X_test = [], [], []
Y_train, Y_valid, Y_test = [], [], []
waveforms = np.array(waveforms)
for i in range(emo_class):
    emotion_indices = [index for index, emotion in enumerate(emotions) if emotion == i]
    np.random.seed(68)
    emotion_indices = np.random.permutation(emotion_indices)
    dim = len(emotion_indices)
    # 8:1:1划分
    train_indices = emotion_indices[:int(0.8 * dim)]
    valid_indices = emotion_indices[int(0.8 * dim):int(0.9 * dim)]
    test_indices = emotion_indices[int(0.9 * dim):]

    X_train.append(waveforms[train_indices, :])
    Y_train.append(np.array([i] * len(train_indices), dtype=np.int32))
    X_valid.append(waveforms[valid_indices, :])
    Y_valid.append(np.array([i] * len(valid_indices), dtype=np.int32))
    X_test.append(waveforms[test_indices, :])
    Y_test.append(np.array([i] * len(test_indices), dtype=np.int32))
    train_set.append(train_indices)
    valid_set.append(valid_indices)
    test_set.append(test_indices)
# 同一个集合连接起来
X_train = np.concatenate(X_train, axis=0)
X_valid = np.concatenate(X_valid, axis=0)
X_test = np.concatenate(X_test, axis=0)
Y_train = np.concatenate(Y_train, axis=0)
Y_valid = np.concatenate(Y_valid, axis=0)
Y_test = np.concatenate(Y_test, axis=0)
train_set = np.concatenate(train_set, axis=0)
valid_set = np.concatenate(valid_set, axis=0)
test_set = np.concatenate(test_set, axis=0)


# 这里是划分数据集完成后再提取mfccs
def get_features(wfs, features):
    file_count = 0
    for wf in wfs:
        Mfccs = librosa.feature.mfcc(
            y=wf,
            sr=sample_rate,
            n_mfcc=40,
            n_fft=1024,
            win_length=512,
            window='hamming',
            fmax=sample_rate / 2
        )
        features.append(Mfccs)
        file_count += 1
    return features


features_train, features_valid, features_test = [], [], []
features_train = get_features(X_train, features_train)
features_valid = get_features(X_valid, features_valid)
features_test = get_features(X_test, features_test)


X_train = np.expand_dims(features_train, 1)
X_valid = np.expand_dims(features_valid, 1)
X_test = np.expand_dims(features_test, 1)
Y_train = np.array(Y_train)
Y_valid = np.array(Y_valid)
Y_test = np.array(Y_test)

X_train = X_Scaler(X_train)
X_valid = X_Scaler(X_valid)
X_test = X_Scaler(X_test)
print(f'X_train scaled:{X_train.shape}, Y_train:{Y_train.shape}')
print(f'X_valid scaled:{X_valid.shape}, Y_valid:{Y_valid.shape}')
print(f'X_test scaled:{X_test.shape}, Y_test:{Y_test.shape}')
# X_train scaled:(960, 1, 40, 130), Y_train:(960,)
# X_valid scaled:(120, 1, 40, 130), Y_valid:(120,)
# X_test scaled:(120, 1, 40, 130), Y_test:(120,)

filename = 'lib/features_labels.npy'
with open(filename, 'wb') as f:
    np.save(f, X_train)
    np.save(f, X_valid)
    np.save(f, X_test)
    np.save(f, Y_train)
    np.save(f, Y_valid)
    np.save(f, Y_test)
    np.save(f, train_set)
