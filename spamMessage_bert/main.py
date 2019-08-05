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
parser.add_argument("-e", "--EPOCHS", default=2, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=32, type=int, help="batch size")
args = parser.parse_args()

'''
flyai库中的提供的数据处理方法
传入整个数据训练多少轮，每批次批大小
'''
dataset = Dataset(epochs=args.EPOCHS, batch=args.BATCH,val_batch=512)

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


# region 训练预测模型
myModel = Model(dataset)
myModel.train_model(needInit=False, loadmodelType='embedding', epochs = args.EPOCHS)
# endregion


# region 创建训练保存模型
# inputParams,outputParams,summaryParams = create_model(vocab_size)
#
# init_op = tf.global_variables_initializer()
# saver = tf.train.Saver(max_to_keep=4,)
# with tf.Session() as sess:
#     sess.run(init_op)
#
#     input_x = inputParams['input_x']
#     input_y = inputParams['input_y']
#     keep_prob = inputParams['keep_prob']
#
#     loss = outputParams['loss']
#     accuracy = outputParams['accuracy']
#     train_op = outputParams['train_op']
#
#     merged_summary = summaryParams['merged_summary']
#
#     train_writer = tf.summary.FileWriter(LOG_PATH, sess.graph)
#
#     for j in range(dataset.get_step()):
#         x_train,y_train = dataset.next_train_batch()
#
#
#         fetches = [loss,accuracy,train_op]
#         feed = {input_x:x_train,input_y:y_train,keep_prob:0.8}
#         loss_, accuracy_, _ = sess.run(fetches,feed_dict=feed)
#
#         if j % 1 == 0 or j == dataset.get_step() - 1:
#             x_val, y_val = dataset.next_validation_batch()
#             summary_train = sess.run(merged_summary, feed_dict=feed, )
#             train_writer.add_summary(summary_train, j)
#             summary_val = sess.run([loss, accuracy],
#                                    feed_dict={input_x: x_val, input_y: y_val, keep_prob: 1.0})
#             print('当前批次: {} | 当前训练损失: {} | 当前训练准确率： {} | '
#                   '当前验证集损失： {} | 当前验证集准确率： {}'.format(j, loss_, accuracy_, summary_val[0],
#                                                       summary_val[1]))
#             save_path = saver.save(sess, os.path.join(MODEL_PATH, TENSORFLOW_MODEL_DIR), global_step=dataset.get_step(),)
#             print("Model saved in path: %s" % save_path)
# endregion


# region 加载模型，预测
# with tf.Session() as sess:
#     saver = tf.train.import_meta_graph(os.path.join(MODEL_PATH, TENSORFLOW_MODEL_DIR) + "-6.meta")
#     saver.restore(sess, os.path.join(MODEL_PATH, TENSORFLOW_MODEL_DIR) + '-6')
#
#     x_train, y_train = dataset.get_all_validation_data()
#
#     graph = tf.get_default_graph()
#     graph = sess.graph
#
#     # loss = graph.get_tensor_by_name("loss:0")
#     accuracy = graph.get_tensor_by_name("accuracy/accuracy:0")
#     # train_op = graph.get_tensor_by_name("train_op:0")
#     keep_prob = graph.get_tensor_by_name("keep_prob:0")
#     y_pred_cls = graph.get_tensor_by_name('fc2/y_pred:0')  # 预测类别
#     # y_pred_cls = graph.get_operation_by_name('fc2/y_pred')  # 预测类别
#     input_x = graph.get_tensor_by_name('input_x:0')  # 预测类别
#
#
#     feed = {input_x:x_train,keep_prob:1}
#     predicts = sess.run(y_pred_cls,feed_dict=feed)
#     print(predicts)
# endregion

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

