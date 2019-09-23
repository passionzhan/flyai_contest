# -*- coding: utf-8 -*-
import argparse
import math

import numpy as np
from numpy import random
from flyai.dataset import Dataset
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler

from config import ans_dict
from data_helper import process_ans_batch
from model import QAModel
from utilities import data_split
from path import *
'''
项目中的超参
'''
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=1, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=3, type=int, help="batch size")
parser.add_argument("-vb", "--VAL_BATCH", default=64, type=int, help="val batch size")
args = parser.parse_args()

#  在本样例中， args.BATCH 和 args.VAL_BATCH 大小需要一致
'''
flyai库中的提供的数据处理方法
传入整个数据训练多少轮，每批次批大小
'''
dataset = Dataset(epochs=args.EPOCHS, batch=args.BATCH, val_batch=args.VAL_BATCH)
model = QAModel(dataset)

print("number of train examples:%d" % dataset.get_train_length())
print("number of validation examples:%d" % dataset.get_validation_length())

seq2seqModel = model.seq2seqModel
seq2seqModel.summary()

x_train, y_train, x_val, y_val = data_split(dataset,val_ratio=0.1)
# x_train, _  = x_train
# y_train, _  = y_train
# x_val, _    = x_val
# y_val, _    = y_val
train_len   = x_train.shape[0]
val_len     = x_val.shape[0]

def gen_batch_data(dataset, x,y, batch_size):
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

        x_batch, x_batch_len = dataset.processor_x(x_batch)
        y_batch, y_batch_len = dataset.processor_y(y_batch)
        y_batch = process_ans_batch(y_batch, ans_dict, int(sorted(list(y_batch_len), reverse=True)[0]))
        y_input_batch = y_batch[:, 0:y_batch.shape[1] - 1]
        y_output_batch = y_batch[:, 1:y_batch.shape[1]]
        y_output_batch = np.reshape(y_output_batch,
                                    (y_output_batch.shape[0],y_output_batch.shape[1],1))
        yield [x_batch, y_input_batch], y_output_batch

steps_per_epoch = math.ceil(train_len / args.BATCH)
val_steps_per_epoch = math.ceil(val_len / args.BATCH)
print("real number of train examples:%d" % train_len)
print("real number of validation examples:%d" % x_val.shape[0])
print("steps_per_epoch:%d" % steps_per_epoch)
print("val_steps_per_epoch:%d" % val_steps_per_epoch)

train_gen   = gen_batch_data(dataset,x_train,y_train,args.BATCH)
val_gen     = gen_batch_data(dataset,x_val,y_val,args.BATCH)

checkpoint = ModelCheckpoint(model.model_path,
                             monitor='val_sparse_categorical_accuracy',
                             save_best_only=True,
                             save_weights_only=True,
                             verbose=1,
                             mode='max')
earlystop = EarlyStopping(patience=1,)
lrs = LearningRateScheduler(lambda epoch, lr, : 0.9*lr, verbose=1)
cbs = [checkpoint, earlystop, lrs]

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)
# 超参数
batch_size = args.BATCH

seq2seqModel.fit_generator(generator=train_gen,
                           steps_per_epoch=int(steps_per_epoch/2),
                           epochs=args.EPOCHS*2,
                           validation_data=val_gen,
                           validation_steps=int(val_steps_per_epoch/2),
                           verbose=1,
                           validation_freq=1,
                           callbacks=cbs)
