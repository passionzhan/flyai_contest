# -*- coding: utf-8 -*-
import argparse
import math

import numpy as np
from numpy import random
from flyai.dataset import Dataset
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler

from data_helper import tokenizer
from model import Model
from utilities import data_split
from path import *
from config import *

def show_result(model):
    s1_que = u'最近一段时间脸上发红起一些像被蚊子咬的那种胞一样很痒但实际不是被蚊子咬的这是什么症状怎么治疗说我季节过敏或者毛细血管扩张想治疗好不知如何是好'
    s1_ans = '看你说的这个症状，那看到是不排除是有过敏方面的原因引起的，这时可以吃上几天抗过敏的药物治疗看看效果也行的还有就是这段时间你饮食方面要清淡些才好，不要吃辛辣刺激性大的食物的，还有就是像海鲜类容易引起过敏的食物暂时也先不要吃才行'
    s2_que = u'我试管移植第13天血值9。95，我已经停药了，可今天第19天我抽血血值又是42了，我这个不会是宫外孕吧，怎么办试管婴儿中我该继续观察血值，还是可以做B超观察了'
    s2_ans = u'你的情况考虑是试管移植的情况你的情况一般情况考虑是进行复查是可以的，估计不会宫外孕的情况'

    for s in [s1_que, s2_que]:
        predict = model.decode_sequence(s)
        predict = tokenizer.decode(predict[0])
        print(predict)

'''
项目中的超参
'''
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=22, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=3, type=int, help="batch size")
parser.add_argument("-vb", "--VAL_BATCH", default=3, type=int, help="val batch size")
args = parser.parse_args()

#  在本样例中， args.BATCH 和 args.VAL_BATCH 大小需要一致
'''
flyai库中的提供的数据处理方法
传入整个数据训练多少轮，每批次批大小
'''
dataset = Dataset(epochs=args.EPOCHS, batch=args.BATCH, val_batch=args.VAL_BATCH)
model = Model(dataset)

print("number of train examples:%d" % dataset.get_train_length())
print("number of validation examples:%d" % dataset.get_validation_length())

bert4nlg_model = model.bert4nlg_model
bert4nlg_model.summary()

x_train, y_train, x_val, y_val = data_split(dataset,val_ratio=0.1)
train_len   = x_train.shape[0]
val_len     = x_val.shape[0]

def padding(x):
    """padding至batch内的最大长度
    """
    ml = max([len(i) for i in x])
    return np.array([i + [0] * (ml - len(i)) for i in x])

def gen_batch_data(x,y, batch_size):
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

    x_batch, y_batch = [], []
    while True:
        bi = i*batch_size
        ei = min(i*batch_size + batch_size,len(indices))
        if ei == len(indices):
            i = 0
        else:
            i += 1

        for idx in range(bi,ei):
            # 确保编码后也不超过max_seq_len
            x_      = x[idx]["que_text"][:max_seq_len-2]
            y_      = y[idx]["ans_text"][:max_seq_len-2]
            x_, y_  = tokenizer.encode(x_, y_)
            x_batch.append(x_)
            y_batch.append(y_)

        x_batch = padding(x_batch)
        y_batch = padding(y_batch)
        yield [x_batch, y_batch], None
        x_batch, y_batch = [], []

steps_per_epoch = math.ceil(train_len / args.BATCH)
val_steps_per_epoch = math.ceil(val_len / args.BATCH)
print("real number of train examples:%d" % train_len)
print("real number of validation examples:%d" % x_val.shape[0])
print("steps_per_epoch:%d" % steps_per_epoch)
print("val_steps_per_epoch:%d" % val_steps_per_epoch)

train_gen   = gen_batch_data(x_train,y_train,args.BATCH)
val_gen     = gen_batch_data(x_val,y_val,args.BATCH)

class myModelCheckpoint(ModelCheckpoint):
    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(myModelCheckpoint, self).__init__(filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1)

    def on_epoch_end(self, epoch, logs=None):
        super(myModelCheckpoint, self).on_epoch_end(epoch, logs)
        show_result(model)

checkpoint = myModelCheckpoint(model.model_path,
                             monitor='val_loss',
                             save_best_only=True,
                             save_weights_only=True,
                             verbose=1,
                             mode='min')
earlystop = EarlyStopping(monitor='val_loss',patience=5,verbose=1,)

def changeLR(epoch, lr):
    if epoch < 10:
        return 0.75*lr
    else:
        return 0.9 * lr

# lrs = LearningRateScheduler(changeLR, verbose=1)
lrs = LearningRateScheduler(lambda epoch, lr, : 0.9*lr, verbose=1)

cbs = [checkpoint, earlystop, lrs]

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)
# 超参数
batch_size = args.BATCH

bert4nlg_model.fit_generator(generator=train_gen,
                             steps_per_epoch=int(steps_per_epoch/2),
                             epochs=args.EPOCHS*2,
                             validation_data=val_gen,
                             validation_steps=int(val_steps_per_epoch/2),
                             verbose=1,
                             validation_freq=1,
                             callbacks=cbs)
