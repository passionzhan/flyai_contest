# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 19:44:02 2017

@author: user
"""

import argparse

import keras
from flyai.dataset import Dataset
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, Activation, MaxPooling2D
from keras.models import Sequential
from model import Model
from path import MODEL_PATH

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
parser.add_argument("-e", "--EPOCHS", default=10, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=32, type=int, help="batch size")
args = parser.parse_args()

'''
flyai库中的提供的数据处理方法
传入整个数据训练多少轮，每批次批大小
'''
dataset = Dataset(epochs=args.EPOCHS, batch=args.BATCH)
model = Model(dataset)

'''
实现自己的网络机构
'''
sqeue = Sequential()

# 第一个卷积层，32个卷积核，大小５x5，卷积模式SAME,激活函数relu,输入张量的大小
sqeue.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu',
                 input_shape=(224, 224, 3)))
sqeue.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu'))
# 池化层,池化核大小２x2
sqeue.add(MaxPool2D(pool_size=(2, 2)))
sqeue.add(Dropout(0.25))
sqeue.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
sqeue.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
sqeue.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
sqeue.add(Dropout(0.25))
# 全连接层,展开操作，
sqeue.add(Flatten())
# 添加隐藏层神经元的数量和激活函数
sqeue.add(Dense(256, activation='relu'))
sqeue.add(Dropout(0.25))
# 输出层
sqeue.add(Dense(6, activation='softmax'))

# 输出模型的整体信息
sqeue.summary()

sqeue.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

'''
dataset.get_step() 获取数据的总迭代次数

'''
best_score = 0
for step in range(dataset.get_step()):
    x_train, y_train = dataset.next_train_batch()
    x_val, y_val = dataset.next_validation_batch()
    history = sqeue.fit(x_train, y_train,
                        batch_size=args.BATCH,
                        verbose=1)
    score = sqeue.evaluate(x_val, y_val, verbose=0)
    if score[1] > best_score:
        best_score = score[1]
        '''
        保存模型
        '''
        model.save_model(sqeue, MODEL_PATH, overwrite=True)
        print("step %d, best accuracy %g" % (step, best_score))
    print(str(step + 1) + "/" + str(dataset.get_step()))
