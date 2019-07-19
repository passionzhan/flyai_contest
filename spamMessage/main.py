#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : main.py
# @Author: Zhan
# @Date  : 7/18/2019
# @Desc  :
import argparse

import tensorflow as tf
from tensorflow import keras
from flyai.dataset import Dataset

from processor import Processor

print('-------------------------------------')
print(tf.__version__)

'''
项目中的超参
'''
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=1, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=64, type=int, help="batch size")
args = parser.parse_args()

'''
flyai库中的提供的数据处理方法
传入整个数据训练多少轮，每批次批大小
'''
dataset = Dataset(epochs=args.EPOCHS, batch=args.BATCH)

vocab_size = Processor().getWordsCount()

# region 模型超参数
e_dim = 300
fc1_dim = 32
# endregion

# region 准备数据
allDataLength = dataset.get_train_length()
print('length of all dev data: %d' % allDataLength)
x, y, x_ , y_  = dataset.get_all_processor_data()

trainLen = (int)(95*allDataLength/100)
x_train = x[0:trainLen]
y_train = y[0:trainLen]
x_val = x[trainLen:]
y_val = y[trainLen:]
# endregion

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, e_dim))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(fc1_dim, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.summary()

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train,
                    y_train,
                    epochs=args.EPOCHS,
                    batch_size=args.BATCH,
                    validation_data=(x_val, y_val),
                    verbose=1)

results = model.evaluate(x_val, y_val)

print(results)

