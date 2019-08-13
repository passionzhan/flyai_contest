# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 19:44:02 2017

@author: user
"""
from sklearn.metrics import f1_score
import numpy as np
import argparse
from flyai.dataset import Dataset
from model import Model
from path import MODEL_PATH
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.applications.resnet50 import ResNet50, preprocess_input
from flyai.utils import remote_helper

# 获取预训练模型路径
path = remote_helper.get_remote_date("https://www.flyai.com/m/v0.2|resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5")

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
dataset = Dataset(epochs=args.EPOCHS, batch=args.BATCH)
model = Model(dataset)
nb_classes = 45
start_lr = 0.001

'''
实现自己的网络结构
'''
inputs = tf.placeholder(shape=(None, 224, 224, 3), dtype=tf.float32, name='inputs')
labels = tf.placeholder(shape=(None, nb_classes), dtype=tf.int32, name='labels')
learning_rate = tf.placeholder(dtype=tf.float32, name='lr')
inputs = preprocess_input(inputs, mode='tf')

resNet50 = ResNet50(include_top=False, weights=None, pooling='avg')
features = resNet50(inputs)
fc_1 = tf.keras.layers.Dense(512, activation='relu')(features)
logits = tf.keras.layers.Dense(nb_classes)(fc_1)
outputs = tf.nn.softmax(logits, name='outputs')

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
acc_value = tf.reduce_mean(tf.keras.metrics.categorical_accuracy(labels, outputs))

saver = tf.train.Saver()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

val_acc = 0

with tf.keras.backend.get_session() as sess:
    sess.run(tf.global_variables_initializer())
    print('load pretrain model...')
    resNet50.load_weights(path)
    print('load done !!!')



    for step in range(dataset.get_step()):  # dataset.get_step() 获取数据的总迭代次数
        x_train, y_train = dataset.next_train_batch()
        _, temp_train_loss, temp_train_acc, temp_train_outputs = sess.run([train_step, loss, acc_value, outputs], feed_dict={inputs: x_train, labels: y_train, learning_rate: lr, K.learning_phase(): 1})
        # 计算训练中的f1 score
        temp_train_pred = np.argmax(temp_train_outputs, axis=-1)
        temp_train_label = np.argmax(y_train, axis=-1)
        temp_train_f1 = f1_score(temp_train_label, temp_train_pred, average='macro')
        print('step: %d/%d, train_loss: %f， train_acc: %f, train_f1: %f'%(step+1, dataset.get_step(), temp_train_loss, temp_train_acc, temp_train_f1))
        if temp_train_loss <= 1:
            lr = start_lr * 0.1
        if step % 50 == 0:
            x_val, y_val = dataset.next_validation_batch()
            temp_val_acc, temp_val_loss, temp_val_outputs = sess.run([acc_value, loss, outputs], feed_dict={inputs: x_val, labels: y_val, K.learning_phase(): 0})
            # 计算校验集中的f1 score
            temp_val_pred = np.argmax(temp_val_outputs, axis=-1)
            temp_val_label = np.argmax(y_val, axis=-1)
            temp_val_f1 = f1_score(temp_val_label, temp_val_pred, average='macro')
            print('--------------- val_loss: %f, val_acc: %f, val_f1: %f'%(temp_val_loss, temp_val_acc, temp_val_f1))
            if temp_val_acc >= val_acc:
                val_acc = temp_val_acc
                # 保存模型
                model.save_model(sess, MODEL_PATH, overwrite=True)
                print('--------------- saved model !!!!')


