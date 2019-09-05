#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : main.py
# @Author: Zhan
# @Date  : 8/19/2019
# @Desc  :

import argparse,os
from functools import reduce
from math import pow,ceil,floor

from flyai.utils import remote_helper
from flyai.dataset import Dataset
import keras
import numpy as np
from numpy import random
from keras.layers import Dense, Dropout
from keras import models
from keras.metrics import categorical_accuracy
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras.utils import plot_model
from keras_applications.densenet import DenseNet201, preprocess_input

from model import Model
from utilities import data_split
from path import MODEL_PATH

# 获取预训练模型路径
# path = remote_helper.get_remote_date("https://www.flyai.com/m/v0.2|resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5")
path = remote_helper.get_remote_date("https://www.flyai.com/m/v0.8|densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5")
# path = r"D:/jack_doc/python_src/flyai/data/SceneClassification_FlyAI_data/v0.8_densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5"

'''
Tensorflow模版项目下载： https://www.flyai.com/python/tensorflow_template.zip
PyTorch模版项目下载： https://www.flyai.com/python/pytorch_template.zip
Keras模版项目下载： https://www.flyai.com/python/keras_template.zip
第一次使用请看项目中的：第一次使用请读我.html文件
常见问题请访问：https://www.flyai.com/question
意见和问题反馈有红包哦！添加客服微信：flyaixzs
'''

'''
项目的超参
'''
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=1, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=2, type=int, help="batch size")
args = parser.parse_args()


'''
flyai库中的提供的数据处理方法
传入整个数据训练多少轮，每批次批大小
'''
print('batch_size: %d, epoch_size: %d'%(args.BATCH, args.EPOCHS))
dataset = Dataset(epochs=args.EPOCHS, batch=args.BATCH, val_batch=6)

model = Model(dataset)

print("number of train examples:%d" % dataset.get_train_length())
print("number of validation examples:%d" % dataset.get_validation_length())

# x_train,y_train,x_val,y_val = dataset.get_all_processor_data()

# region 超参数
n_classes = 45
fc1_dim = 512
# endregion

# region 定义网络结构
kwargs = {'backend':keras.backend,
          'layers': keras.layers,
          'models': keras.models,
          'utils': keras.utils}

densenet201     = DenseNet201(include_top=False, weights=None, pooling='avg', **kwargs)
features        = densenet201.output
fc1             = Dense(fc1_dim, activation='relu',)(features)
fc1_D           = Dropout(0.15,)(fc1)
predictions     = Dense(n_classes, activation='softmax')(fc1_D)

mymodel         = models.Model(inputs=densenet201.input, outputs=predictions)

mymodel.compile(loss='categorical_crossentropy',
                    optimizer=keras.optimizers.Adam(lr=0.001,),
                    metrics=[categorical_accuracy])

# region 打印模型信息
# mymodel.summary()
# plot_model 需要安装pydot and graphviz
# plot_model(mymodel, to_file='mymodel.png')
# endregion

print('load pretrain model...')
densenet201.load_weights(path)
print('load done !!!')

x_train, y_train, x_val, y_val = data_split(dataset,val_ratio=0.1)
train_len   = x_train.shape[0]

def gen_batch_data(dataset, x, y, batch_size):
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
        x_batch = dataset.processor_x(x_data)
        y_batch = dataset.processor_y(y_data)
        yield x_batch, y_batch

checkpoint = ModelCheckpoint(model.model_path,
                             monitor='val_categorical_accuracy',
                             save_best_only=True,
                             save_weights_only=False,
                             verbose=1,
                             mode='max',
                             period=1,)
earlystop = EarlyStopping(monitor='val_categorical_accuracy', verbose=1, patience=150,)
lrs = LearningRateScheduler(lambda epoche, lr: pow(0.9,epoche//50)*lr, verbose=1)
cbs = [checkpoint, earlystop, lrs]

train_generator = gen_batch_data(dataset,x_train,y_train,args.BATCH)
val_generator   = gen_batch_data(dataset,x_val,y_val,args.BATCH)

steps_per_epoch = ceil(train_len / (100 * args.BATCH))
# steps_per_epoch = 5
# steps_per_epoch =
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

mymodel.fit_generator(generator=train_generator, steps_per_epoch=steps_per_epoch,
                      epochs=(100 * args.EPOCHS), validation_data=val_generator,
                      validation_steps=25,verbose=1,
                      validation_freq=1,
                      callbacks=cbs)