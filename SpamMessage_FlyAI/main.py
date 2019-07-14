# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 19:44:02 2017

@author: user
"""

import argparse
import tensorflow as tf
from flyai.dataset import Dataset
from model import Model
from data_helper import *
from path import MODEL_PATH, LOG_PATH

print('-------------------------------------')
print(tf.__version__)

'''
Tensorflow模版项目下载： https://www.flyai.com/python/tensorflow_template.zip
PyTorch模版项目下载： https://www.flyai.com/python/pytorch_template.zip
Keras模版项目下载： https://www.flyai.com/python/keras_template.zip
第一次使用请看项目中的：第一次使用请读我.html文件
常见问题请访问：https://www.flyai.com/question
意见和问题反馈有红包哦！添加客服微信：flyaixzs
'''

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
dataset = Dataset(epochs=args.EPOCHS, batch=args.BATCH)
model = Model(dataset)

word_dict, word_dict_res = load_dict()
vocab_size = max(word_dict.values()) + 1

# 超参
embedding_dim = 64      # 嵌入层大小
dnn_dim = 128           # Dense层大小
max_seq_len = 128       # 最大句长
num_filters = 64        # 卷积核数目
kernel_size = 5         # 卷积核尺寸
learning_rate = 1e-3    # 学习率
numclass = 2            # 类别数


# 传值空间
input_x = tf.placeholder(tf.int32, shape=[None, max_seq_len], name='input_x')
input_y = tf.placeholder(tf.int32, shape=[None], name='input_y')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')


# define embedding layer
with tf.variable_scope('embedding'):
    # 标准正态分布初始化
    input_embedding = tf.Variable(
        tf.truncated_normal(shape=[vocab_size, embedding_dim], stddev=0.1), name='encoder_embedding')

with tf.name_scope("cnn"):
    # CNN layer
    x_input_embedded = tf.nn.embedding_lookup(input_embedding, input_x)
    conv = tf.layers.conv1d(x_input_embedded, num_filters, kernel_size, name='conv')
    # global max pooling layer
    pooling = tf.reduce_max(conv, reduction_indices=[1])

with tf.name_scope("score"):
    # 全连接层，后面接dropout以及relu激活
    fc = tf.layers.dense(pooling, dnn_dim, name='fc1')
    fc = tf.contrib.layers.dropout(fc, keep_prob)
    fc = tf.nn.relu(fc)

    # 分类器
    logits = tf.layers.dense(fc, numclass, name='fc2')
    y_pred_cls = tf.argmax(tf.nn.softmax(logits), 1, name='y_pred')  # 预测类别

with tf.name_scope("optimize"):
    # 将label进行onehot转化
    one_hot_labels = tf.one_hot(input_y, depth=numclass, dtype=tf.float32)
    # 损失函数，交叉熵
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_labels)
    loss = tf.reduce_mean(cross_entropy)
    # 优化器
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

with tf.name_scope("accuracy"):
    # 准确率
    correct_pred = tf.equal(tf.argmax(one_hot_labels, 1), y_pred_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='acc')

with tf.name_scope("summary"):
    tf.summary.scalar("loss", loss)
    tf.summary.scalar("accuracy", accuracy)
    merged_summary = tf.summary.merge_all()

best_score = 0
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter(LOG_PATH, sess.graph)

    # dataset.get_step() 获取数据的总迭代次数
    for step in range(dataset.get_step()):
        x_train, y_train = dataset.next_train_batch()
        x_val, y_val = dataset.next_validation_batch()

        fetches = [loss, accuracy, train_op]
        feed_dict = {input_x: x_train, input_y: y_train, keep_prob: 0.9}
        loss_, accuracy_, _ = sess.run(fetches, feed_dict=feed_dict)

        valid_acc = sess.run(accuracy, feed_dict={input_x: x_val, input_y: y_val, keep_prob: 1.0})
        summary = sess.run(merged_summary, feed_dict=feed_dict)
        train_writer.add_summary(summary, step)

        cur_step = str(step + 1) + "/" + str(dataset.get_step())
        print('The Current step per total: {} | The Current loss: {} | The Current ACC: {} |'
              ' The Current Valid ACC: {}'.format(cur_step, loss_, accuracy_, valid_acc))
        if step % 100 == 0:  # 每隔100个step存储一次model文件
            model.save_model(sess, MODEL_PATH, overwrite=True)