#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : main.py
# @Author: Zhan
# @Date  : 8/13/2019
# @Desc  :
import argparse

import numpy as np
from sklearn.metrics import f1_score
import tensorflow as tf
from flyai.utils import remote_helper
from flyai.dataset import Dataset
# import keras.backend as K
import tensorflow.keras.backend as K
from tensorflow.keras.applications.densenet import DenseNet201, preprocess_input
# import keras.backend as K
# from keras.applications.densenet import DenseNet201, preprocess_input
# from keras.applications.densenet import DenseNet201, preprocess_input

from path import MODEL_PATH, LOG_PATH
from model import Model

# 获取预训练模型路径
# path = remote_helper.get_remote_date("https://www.flyai.com/m/v0.2|resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5")
path = remote_helper.get_remote_date("https://www.flyai.com/m/v0.8|densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5")
# path = r"D:/jack_doc/python_src/flyai/data/SceneClassification_FlyAI_data/v0.8-densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5"

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

# region 定义输入变量
x_inputs    = tf.placeholder(shape=(None, 224, 224, 3), dtype=tf.float32, name='x_inputs')
y_inputs    = tf.placeholder(shape=(None, n_classes), dtype=tf.float32, name='y_inputs')
# lr          = tf.placeholder(dtype=tf.float32, name='lr')
inputs      = preprocess_input(x_inputs,)
# endregion

# region 定义网络结构
densenet201    = DenseNet201(include_top=False, weights=None, pooling='avg')
features    = densenet201(inputs)
fc1         = tf.layers.Dense(fc1_dim, activation='relu')(features)
fc2         = tf.layers.Dense(n_classes,)(fc1)
logits      = tf.nn.softmax(fc2)
pred_y      = tf.argmax(logits, axis=-1, name='pred_y')

loss        = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_inputs,logits=fc2))

# keras 的BatchNormalization当在tensorflow的session中使用时，必须将其手动加入到更新操作集合中
ops = tf.get_default_graph().get_operations()
update_ops = [op for op in ops if ("AssignMovingAvg" in op.name and op.type == "AssignSubVariableOp")]
for op in update_ops:
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, op)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimize    = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss,)
# categorical_accuracy 函数自带类别转换。
accuracy    = tf.reduce_mean(tf.keras.metrics.categorical_accuracy(y_inputs, logits))
# endregion

# saver = tf.train.Saver(var_list = tf.global_variables())
max_val_acc = 0
globals_f1 = 0

with tf.keras.backend.get_session() as sess:
    sess.run(tf.global_variables_initializer())
    print('load pretrain model...')
    densenet201.load_weights(path)
    print('load done !!!')

    # 利用tensorboard查看网络结构
    # writer = tf.summary.FileWriter(LOG_PATH, sess.graph)

    for i in range(dataset.get_step()):
        x_train,y_train = dataset.next_train_batch()
        fetches = [optimize, loss, pred_y, accuracy]
        _, train_loss, train_pred, train_acc = sess.run(fetches,
                 feed_dict={x_inputs:x_train,y_inputs:y_train,K.learning_phase():1},)

        temp_train_f1 = f1_score(np.argmax(y_train,axis=-1), train_pred, average='macro')

        # if i % 50 == 0:
        #     print('step: %d/%d, train_loss: %f， train_acc: %f, train_f1: %f'
        #           %(i+1, dataset.get_step(), train_loss, train_acc, temp_train_f1))

        if i% 100 == 0 or i == dataset.get_step() - 1:
            print('step: %d/%d, train_loss: %f， train_acc: %f, train_f1: %f'
                      %(i+1, dataset.get_step(), train_loss, train_acc, temp_train_f1))
            x_val, y_val = dataset.next_validation_batch()
            val_pred, val_loss, val_acc = sess.run([pred_y, loss, accuracy],
                     feed_dict={x_inputs: x_val, y_inputs: y_val, K.learning_phase(): 0}, )

            temp_val_f1 = f1_score(np.argmax(y_val, axis=-1), val_pred, average='macro')
            print('step: %d/%d, val_loss: %f， val_acc: %f, val_f1: %f'
                  %(i+1, dataset.get_step(), val_loss, val_acc, temp_val_f1))
            if max_val_acc < val_acc or (max_val_acc == val_acc and globals_f1 < temp_val_f1):
                max_val_acc, globals_f1 = val_acc, temp_val_f1
                ### 加入模型保存代码
                # if i == dataset.get_step() - 1:
                model.save_model(sess,MODEL_PATH,overwrite=True)












