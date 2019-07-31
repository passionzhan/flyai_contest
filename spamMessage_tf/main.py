#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : main.py
# @Author: Zhan
# @Date  : 7/18/2019
# @Desc  :
import argparse

import tensorflow as tf
from flyai.dataset import Dataset
# from dataset import Dataset

from processor import Processor
from model import *

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
# dataset = Dataset(train_batch=args.BATCH, val_batch=128, )
dataset = Dataset(epochs=args.EPOCHS, batch=args.BATCH)

vocab_size = Processor().getWordsCount()

# region 准备数据
# allDataLength = dataset.get_train_length()
# print('length of all dev data: %d' % allDataLength)
# x, y, x_ , y_  = dataset.get_all_processor_data()

# trainLen = (int)(95*allDataLength/100)
# x_train = x[0:trainLen]
# y_train = y[0:trainLen]
# x_val = x[trainLen:]
# y_val = y[trainLen:]

# x_train = x
# y_train = y
# x_val = x_
# y_val = y_
# endregion


myModel = Model(dataset)
myModel.train_model(needInit=True, epochs = args.EPOCHS)


# save_callback = tf.keras.callbacks.ModelCheckpoint(myModel.model_path,
#                                                    save_weights_only=True,
#                                                    verbose=1,
#                                                    period=5)
# history = myModel.dpNet.fit(x_train,
#                             y_train,
#                             epochs=args.EPOCHS,
#                             batch_size=args.BATCH,
#                             callbacks=[save_callback,],
#                             validation_data=(x_val, y_val),
#                             verbose=1)

# results = myModel.dpNet.evaluate(x_val, y_val)
#
# print(results)

