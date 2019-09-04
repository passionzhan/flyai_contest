# -*- coding: utf-8 -*
import argparse
import math
import os

import numpy as np
from numpy import random
from flyai.dataset import Dataset
from keras.callbacks import ModelCheckpoint, EarlyStopping

from model import Model
import config
from path import MODEL_PATH
from utilities import data_split
# 超参
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=16, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=64, type=int, help="batch size")
args = parser.parse_args()
# 数据获取辅助类
dataset = Dataset(epochs=args.EPOCHS, batch=args.BATCH,val_batch=args.BATCH)
# 模型操作辅助类
model = Model(dataset)

# 必须使用该方法下载模型，然后加载
# path = remote_helper.get_remote_date("https://www.flyai.com/m/chinese_L-12_H-768_A-12.zip")
# pre_trained_path = r'D:/jack_doc/python_src/flyai/chinese_L-12_H-768_A-12'

print("number of train examples:%d" % dataset.get_train_length())
print("number of validation examples:%d" % dataset.get_validation_length())

'''
keras: bi-LSTM+CRF
'''

# 得到训练和测试的数据
TIME_STEP       = config.max_sequence      # 每个句子的最大长度和time_step一样,为了避免混淆model中全部用time_step表示。
TAGS_NUM        = config.label_len

#
# texts = ['中 美 贸 易 战', '中 国 人 民 解 放 军 于 今 日 在 东 海 举 行 实 弹 演 习']
# embeddings = extract_embeddings(pre_trained_path, texts)

ner_model = model.ner_model
ner_model.summary()

x_train, y_train, x_val, y_val = data_split(dataset,val_ratio=0.1)
x_train     = dataset.processor_x(x_train)
x_val       = dataset.processor_x(x_val)
y_train     = dataset.processor_y(y_train)
y_val       = dataset.processor_y(y_val)
train_len   = x_train.shape[0]

# print('x_train')
def gen_batch_data(x,y,batch_size):
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
        x_data = x[bi:ei]
        y_data = y[bi:ei]
        max_seq_length = max(map(len, x_data))
        # print('bi %d:' % bi)
        # print('ei %d:' % ei)
        # if x_smp.shape[0] < TIME_STEP:
        # x_batch = np.asarray([list(x_smp[:]) + (TIME_STEP - x_smp.shape[0]) * [config.src_padding] if x_smp.shape[0]<TIME_STEP else
        #                       list(x_smp[:])[0:TIME_STEP] for x_smp in x_data])
        # y_batch = np.asarray([list(y_smp[:]) + (TIME_STEP - y_smp.shape[0]) * [TAGS_NUM - 1] if y_smp.shape[0]<TIME_STEP else
        #                       list(y_smp[:])[0:TIME_STEP] for y_smp in y_data])

        # Embedding 层 mask_zero = True，所以 x 用 0 补齐
        x_batch = np.asarray([list(x_smp[:]) + (max_seq_length - x_smp.shape[0]) * [0] for x_smp in x_data])
        y_batch = np.asarray([list(y_smp[:]) + (max_seq_length - y_smp.shape[0]) * [TAGS_NUM - 1] for y_smp in y_data])

        tmpShape = y_batch.shape
        y_batch = y_batch.reshape((tmpShape[0], tmpShape[1], 1))
        yield x_batch, y_batch


steps_per_epoch = math.ceil(train_len / args.BATCH)
print("real number of train examples:%d" % train_len)
print("real number of validation examples:%d" % x_val.shape[0])
print("steps_per_epoch:%d" % steps_per_epoch)

train_gen   = gen_batch_data(x_train,y_train,args.BATCH)
val_gen     = gen_batch_data(x_val,y_val,args.BATCH)


# checkpoint = ModelCheckpoint(model.model_path,
#                              monitor='train_loss',
#                              save_best_only=True,
#                              save_weights_only=True,
#                              verbose=1,
#                              mode='max')
checkpoint = ModelCheckpoint(model.model_path,
                             monitor='val_crf_accuracy',
                             save_best_only=True,
                             save_weights_only=True,
                             verbose=1,
                             mode='max')
earlystop = EarlyStopping(patience=10,)

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)
# ner_model.fit_generator(generator=train_gen, steps_per_epoch=steps_per_epoch,
#                         epochs=args.EPOCHS,
#                         callbacks=[checkpoint, earlystop])

ner_model.fit_generator(generator=train_gen, steps_per_epoch=steps_per_epoch,
                        epochs=args.EPOCHS,validation_data=val_gen, validation_steps= 5,
                        callbacks=[checkpoint, earlystop])

# # max_val_acc, min_loss = 0, float('inf')
# for i in range(dataset.get_step()):
#     x_train, y_train = dataset.next_train_batch()
#     # padding
#
#     ner_model.train_on_batch(x_train,y_train)
#
#     if i % 50 == 0 or i == dataset.get_step() - 1:
#
#         x_val, y_val = dataset.next_validation_batch()
#         # padding
#         x_val = np.asarray([list(x[:]) + (TIME_STEP - len(x)) * [config.src_padding] for x in x_val])
#         y_val = np.asarray([list(y[:]) + (TIME_STEP - len(y)) * [TAGS_NUM - 1] for y in y_val])
#         y_val = y_train.reshape(y_val.shape[0], y_val.shape[1], 1)
#         val_loss_and_metrics = ner_model.evaluate(x_val, y_val, verbose=0)
#         cur_loss = val_loss_and_metrics[0]
#         cur_acc = val_loss_and_metrics[1]
#
#
#         print('step: %d/%d, val_loss: %f， val_acc: %f'
#               % (i + 1, dataset.get_step(), cur_loss, cur_acc,))
#         # val_loss_and_metrics[1],))
#
#         if max_val_acc < cur_acc \
#                 or (max_val_acc == cur_acc and min_loss > cur_loss):
#             max_val_acc, min_loss = cur_acc, cur_loss
#             print('max_acc: %f, min_loss: %f' % (max_val_acc, min_loss))
#             model.save_model(ner_model, overwrite=True)