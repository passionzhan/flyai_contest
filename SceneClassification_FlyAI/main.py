#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : main.py
# @Author: Zhan
# @Date  : 8/19/2019
# @Desc  :

import argparse
from functools import reduce

from flyai.utils import remote_helper
from flyai.dataset import Dataset
import keras
import numpy as np
from keras.layers import Dense
from keras import models
from keras.metrics import categorical_accuracy
from keras.utils import plot_model
from keras_applications.densenet import DenseNet201, preprocess_input
from model import Model

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
dataset = Dataset(epochs=args.EPOCHS, batch=args.BATCH, val_batch=6)
model = Model(dataset)

print("number of train examples:%d" % dataset.get_train_length())
print("number of validation examples:%d" % dataset.get_validation_length())

# for i in range(100):
#     val_x,val_y = dataset.next_validation_batch()
#     print(val_x.shape[0])

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
predictions      = Dense(n_classes, activation='softmax')(fc1)

mymodel         = models.Model(inputs=densenet201.input, outputs=predictions)

mymodel.compile(loss='categorical_crossentropy',
                    optimizer=keras.optimizers.Adam(lr=0.0005,),
                    metrics=[categorical_accuracy])

# region 打印模型信息
# mymodel.summary()
# plot_model 需要安装pydot and graphviz
# plot_model(mymodel, to_file='mymodel.png')
# endregion

print('load pretrain model...')
densenet201.load_weights(path)
print('load done !!!')

max_val_acc = 0
min_loss = float('inf')
iCount = 0

for i in range(dataset.get_step()):
    x_train, y_train = dataset.next_train_batch()
    x_train = preprocess_input(x_train,**kwargs)
    mymodel.train_on_batch(x_train, y_train)

    if i % 80 == 0 or i == dataset.get_step() - 1:
        iter_num = dataset.get_validation_length()/6

        # 直接丢弃不够一次循环的验证数据
        if iCount + 144 > iter_num:
            for iLoop in range(iCount+1,int(iter_num+1)):
                dataset.next_validation_batch()
            iCount = 0
            continue

        for iLoop in range(18):
            extra_x_train = np.zeros(shape=(36,224,224,3), dtype=np.uint8)
            extra_y_train = np.zeros(shape=(36,n_classes), dtype=np.uint8)

            for j in range(6):
                x_val, y_val = dataset.next_validation_batch()
                iCount += 1
                for ii in range(x_val.shape[0]):
                    extra_x_train[ii] = x_val[ii]
                    extra_y_train[ii] = y_val[ii]

            extra_x_train = preprocess_input(extra_x_train, **kwargs)
            train_loss_and_metrics = mymodel.train_on_batch(extra_x_train, extra_y_train)
            if iLoop == 17:
                # train_loss_and_metrics = mymodel.evaluate(extra_x_train, extra_y_train,batch_size=36,verbose=0)
                print('step: %d/%d, train_loss: %f， train_acc: %f, '
                      % (i + 1, dataset.get_step(), train_loss_and_metrics[0],
                         train_loss_and_metrics[1]))

        val_acc = []
        val_loss = []
        for iLoop in range(6 * 6):
            # 此处获取的x_val样本数为dataset 的 val_batch == 6
            x_val, y_val = dataset.next_validation_batch()
            iCount += 1
            x_val = preprocess_input(x_val, **kwargs)
            val_loss_and_metrics = mymodel.evaluate(x_val, y_val,verbose=0)
            val_loss.append(val_loss_and_metrics[0])
            val_acc.append(val_loss_and_metrics[1])

        cur_acc = reduce(lambda x, y: x + y, val_acc) / len(val_acc)
        cur_loss = reduce(lambda x, y: x + y, val_loss) / len(val_loss)


        # val_Precision = val_loss_and_metrics[2]
        # val_Reacll  = val_loss_and_metrics[3]
        # if val_Precision == 0: val_Precision = 1e-10
        # if val_Reacll == 0: val_Reacll = 1e-10
        # val_F1 = 2 * (val_Precision * val_Reacll) / (val_Precision + val_Reacll)


        print('step: %d/%d, val_loss: %f， val_acc: %f'
              % (i + 1, dataset.get_step(), cur_loss, cur_acc,))
                 # val_loss_and_metrics[1],))

        if max_val_acc < cur_acc \
                or (max_val_acc == cur_acc and min_loss > cur_loss):
            max_val_acc, min_loss = cur_acc, cur_loss
            print('max_acc: %f, min_loss: %f' % (max_val_acc, min_loss))
            model.save_model(mymodel,overwrite=True)