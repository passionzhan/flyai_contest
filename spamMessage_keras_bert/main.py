#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : main.py
# @Author: Zhan
# @Date  : 7/18/2019
# @Desc  :
import argparse
import math

import numpy as np
from numpy import random
import tensorflow as tf
from flyai.dataset import Dataset
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler

from model import *
from utilities import data_split

print('-------------------------------------')
print(tf.__version__)

'''
项目中的超参，输入参数
'''
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=1, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=2, type=int, help="batch size")
args = parser.parse_args()

'''
flyai库中的提供的数据处理方法
传入整个数据训练多少轮，每批次批大小
'''
dataset = Dataset(epochs=args.EPOCHS, batch=args.BATCH,val_batch=args.BATCH)
print("number of train examples:%d" % dataset.get_train_length())
print("number of validation examples:%d" % dataset.get_validation_length())

def gen_batch_data(dataset, x,y,batch_size, max_seq_len=256):
    '''
    批数据生成器
    :param x:
    :param y:
    :param batch_size:
    :return:
    '''
    indices = np.arange(x.shape[0])
    random.shuffle(indices)
    x = x[indices]
    y = y[indices]
    i = 0
    while True:
        bi = i*batch_size
        ei = min(i*batch_size + batch_size,len(indices))
        if ei == len(indices):
            i = 0
        else:
            i += 1
        x_batch = x[bi:ei]
        y_batch = y[bi:ei]

        # processor_x   返回的是list 构成的np.array
        x_batch = dataset.processor_x(x_batch)
        y_batch = dataset.processor_y(y_batch)

        x_batch_ids, x_batch_mask, x_batch_seg = conver2Input(x_batch, max_seq_len=max_seq_len)

        yield [x_batch_ids, x_batch_mask], y_batch

# region 训练预测模型
myModel = Model(dataset)
spamModel = myModel.spam_model
spamModel.summary()

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

x_train, y_train, x_val, y_val = data_split(dataset,val_ratio=0.03)
# 698399 样本训练  21600  验证
train_len = x_train.shape[0]
steps_per_epoch = math.ceil(train_len / args.BATCH)
print("real number of train examples:%d" % train_len)
print("real number of validation examples:%d" % x_val.shape[0])
print("steps_per_epoch:%d" % steps_per_epoch)

gen_train_batch = gen_batch_data(dataset, x_train, y_train, args.BATCH)
gen_val_batch = gen_batch_data(dataset, x_val, y_val, args.BATCH)

checkpoint = ModelCheckpoint(myModel.model_path,
                             monitor='val_acc',
                             save_best_only=True,
                             save_weights_only=True,
                             verbose=1,
                             mode='max')
earlystop = EarlyStopping(patience=5,)
lrs = LearningRateScheduler(lambda epoch, lr, : 0.9*lr, verbose=1)

spamModel.fit_generator(generator=gen_train_batch, steps_per_epoch=steps_per_epoch,
                        epochs=args.EPOCHS, validation_data=gen_val_batch, validation_steps=steps_per_epoch//20,
                        validation_freq=1,
                        callbacks=[checkpoint, earlystop,lrs])
# endregion