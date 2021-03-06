#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : main_tf_keras.py
# @Author: Zhan
# @Date  : 8/21/2019
# @Desc  :

import argparse

from flyai.utils import remote_helper
from flyai.dataset import Dataset
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import models
from tensorflow.keras.metrics import categorical_accuracy, Precision, Recall
# from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.densenet import DenseNet201, preprocess_input
from model_tf_keras import Model

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
parser.add_argument("-b", "--BATCH", default=1, type=int, help="batch size")
args = parser.parse_args()

'''
flyai库中的提供的数据处理方法
传入整个数据训练多少轮，每批次批大小
'''
print('batch_size: %d, epoch_size: %d'%(args.BATCH, args.EPOCHS))
dataset = Dataset(epochs=args.EPOCHS, batch=args.BATCH, val_batch=32)
model = Model(dataset)

print("number of train examples:%d" % dataset.get_train_length())
print("number of validation examples:%d" % dataset.get_validation_length())

# region 超参数
n_classes = 45
fc1_dim = 512
# endregion

# region 定义网络结构

densenet201     = DenseNet201(include_top=False, weights=None, pooling='avg',)
features        = densenet201.output
fc1             = Dense(fc1_dim, activation='relu',)(features)
predictions      = Dense(n_classes, activation='softmax')(fc1)

mymodel         = models.Model(inputs=densenet201.input, outputs=predictions)

mymodel.compile(loss='categorical_crossentropy',
                    optimizer=Adam(lr=0.001,),
                    metrics=[categorical_accuracy,Precision(),Recall()])

# region 打印模型信息
# mymodel.summary()
# plot_model 需要安装pydot and graphviz
# plot_model(mymodel, to_file='mymodel.png')
# endregion

print('load pretrain model...')
densenet201.load_weights(path)
print('load done !!!')

max_val_acc = 0
globals_f1 = 0

for i in range(dataset.get_step()):
    x_train, y_train = dataset.next_train_batch()
    x_train = preprocess_input(x_train,)
    mymodel.train_on_batch(x_train, y_train)

    if i % 100 == 0 or i == dataset.get_step() - 1:
        x_val, y_val = dataset.next_validation_batch()
        x_val = preprocess_input(x_val,)
        train_batch = x_train.shape[0]
        val_batch = x_val.shape[0]
        train_loss_and_metrics = mymodel.evaluate(x_train, y_train, batch_size = train_batch)
        val_loss_and_metrics = mymodel.evaluate(x_val, y_val,batch_size = val_batch)
        val_Precision = val_loss_and_metrics[2]
        val_Reacll  = val_loss_and_metrics[3]
        if val_Precision == 0: val_Precision = 1e-10
        if val_Reacll == 0: val_Reacll = 1e-10
        val_F1 = 2 * (val_Precision * val_Reacll) / (val_Precision + val_Reacll)
        print('step: %d/%d, train_loss: %f， train_acc: %f'
              % (i + 1, dataset.get_step(), train_loss_and_metrics[0],
                 train_loss_and_metrics[1]))

        print('step: %d/%d, val_loss: %f， val_acc: %f, val_f1: %f'
              % (i + 1, dataset.get_step(), val_loss_and_metrics[0],
                 val_loss_and_metrics[1],val_F1))

        if max_val_acc < val_loss_and_metrics[1] \
                or (max_val_acc == val_loss_and_metrics[1] and globals_f1 < val_F1):
            max_val_acc, globals_f1= val_loss_and_metrics[1], val_F1
            model.save_model(mymodel,overwrite=True)