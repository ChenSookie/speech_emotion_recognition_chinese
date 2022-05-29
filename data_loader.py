# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Sookie
# @File    :  LSTM模型和cnn模型用到的特征向量数据格式
import fnmatch
import os
import time
import joblib
import librosa
import numpy as np

# data, sampling_rate = librosa.load('CASIA_database/actor_1/fear/206.wav')
# sampling_rate   ------22050
path = 'CASIA_database'
lst = []
start_time = time.time()
actors = os.listdir(path)
actors = fnmatch.filter(actors, "actor_*")
waveforms = []
emotions = []
emotions_dic = {'angry': 0, 'fear': 1, 'happy': 2, 'neutral': 3, 'sad': 4, 'surprise': 5}
dirs = fnmatch.filter(actors, "actor_*")
# 加载音频数据，提取mfccs
for actor in actors:
    dirs = os.listdir(path + '/' + actor)
    for i in dirs:
        print("开始加载：", i)
        wav_path = path + "/" + actor + "/" + i
        print(wav_path)
        files = os.listdir(wav_path)
        for j in fnmatch.filter(files, "*.wav"):  # 这里跟transformer模型的处理不一样
            X, sample_rate = librosa.load(wav_path + "/" + j, res_type='kaiser_fast')
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            label = emotions_dic[i]
            arr = mfccs, label
            lst.append(arr)


print("--- Data loaded. Loading time: %s seconds ---" % (time.time() - start_time))
# 2--------------------------------------
X, y = zip(*lst)
X = np.asarray(X)
y = np.asarray(y)
print(X.shape, y.shape)
X_name = 'X.joblib'
y_name = 'y.joblib'
save_dir = 'lib'
savedX = joblib.dump(X, os.path.join(save_dir, X_name))
savedy = joblib.dump(y, os.path.join(save_dir, y_name))