# -*- coding: utf-8 -*
import os

import tensorflow as tf
from flyai.model.base import Base
from tensorflow import keras
from tensorflow.python.saved_model import tag_constants

from path import MODEL_PATH, LOG_PATH
from processor import Processor
from dataset import Dataset

TENSORFLOW_MODEL_DIR = "dpNet.ckpt"

def create_model(vocab_size, ):

    # region 模型超参数
    e_dim = 200
    filters_num = 256
    kernel_size = 5
    max_seq_len = 128

    fc1_dim = 64
    # fc2_dim = 32
    drop_ratio = 0.2
    learning_rate = 0.01
    # 传值空间
    input_x = tf.placeholder(tf.int64, shape=[None, max_seq_len], name='input_x')
    input_y = tf.placeholder(tf.int64, shape=[None], name='input_y')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    # 定义嵌入层
    with tf.variable_scope("embedding"):
        input_embedding = tf.Variable(
            tf.truncated_normal(shape=[vocab_size, e_dim], stddev=0.1),
            name='encoder_embedding',)
        input_embedded = tf.nn.embedding_lookup(input_embedding,input_x)

    #  定义卷积层
    with tf.variable_scope("cnn"):
        filters = tf.Variable(tf.truncated_normal(shape=[kernel_size, e_dim, filters_num], stddev=0.1),
                              name='filters',)
        cnn = tf.nn.conv1d(input_embedded,filters,1,padding='SAME',name="cnn")
        pool = tf.reduce_max(cnn,axis=[1])

    # 定义全连接层
    with tf.variable_scope("fc1"):
        # fc1 = tf.layers.Dense()

        fc1 = tf.get_variable(shape=[fc1_dim,filters_num],dtype=tf.float32,initializer=tf.initializers.he_normal(),name="fc1")
        bias1 = tf.Variable(tf.zeros(shape=[fc1_dim,]))
        fc1_mid = tf.matmul(pool,fc1,transpose_b=True) + bias1
        fc1_drop = tf.nn.dropout(fc1_mid,keep_prob=keep_prob)
        fc1_rst =  tf.nn.relu(fc1_drop)

    # 定义全连接层2 + 分类层
    with tf.variable_scope("fc2"):
        fc2 = tf.get_variable(shape=[2,fc1_dim],dtype=tf.float32,initializer=tf.initializers.he_normal(),name="fc2")
        bias2 = tf.Variable(tf.zeros(shape=[2,]))
        fc2_rst = tf.matmul(fc1_rst,fc2,transpose_b=True) + bias2

        # 分类器
        y_pred_cls = tf.argmax(tf.nn.softmax(fc2_rst), 1, name='y_pred')  # 预测类别

    with tf.variable_scope("optimize"):
        # 将label进行onehot转化
        one_hot_labels = tf.one_hot(input_y, depth=2, dtype=tf.float32)
        # 损失函数，交叉熵
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=fc2_rst, labels=one_hot_labels)
        loss = tf.reduce_mean(cross_entropy)
        # 优化器
        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    with tf.variable_scope("accuracy"):
        correct_pred = tf.equal(input_y, y_pred_cls)
        accuracy = tf.reduce_mean(tf.cast(correct_pred,dtype=tf.float32),name="accuracy")

    with tf.name_scope("summary"):
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("accuracy", accuracy)
        merged_summary = tf.summary.merge_all()


    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        train_writer = tf.summary.FileWriter(LOG_PATH, sess.graph)

        dataset = Dataset(train_batch=128, val_batch=64, split_ratio = 0.9,)
        epochs = 1

        saver = tf.train.Saver()
        for i in range(epochs):
            for j in range(dataset.step):
                x_train, y_train = dataset.next_train_batch()

                fetches = [loss, accuracy, train_op]

                feed_dict = {input_x: x_train, input_y: y_train, keep_prob: 0.9}
                loss_, accuracy_, _ = sess.run(fetches, feed_dict=feed_dict)

                if j % 10 == 0:
                    x_val, y_val = dataset.next_val_batch()
                    summary_train = sess.run(merged_summary, feed_dict=feed_dict, )
                    train_writer.add_summary(summary_train,i*dataset.step + j)
                    summary_val = sess.run([loss, accuracy], feed_dict={input_x: x_val, input_y: y_val, keep_prob: 1.0})
                    print('当前批次/代数: {}/{} | 当前训练损失: {} | 当前训练准确率： {} | '
                          '当前验证集损失： {} | 当前验证集准确率： {}'.format(j, i, loss_, accuracy_, summary_val[0],summary_val[1]))
                    save_path = saver.save(sess, "/tmp/model.ckpt")
                    print("Model saved in path: %s" % save_path)



class Model(Base):
    def __init__(self, data,):
        self.data = data
        self.model_path = os.path.join(MODEL_PATH, TENSORFLOW_MODEL_DIR)
        self.vocab_size = Processor().getWordsCount()
        self.dpNet = create_model(self.vocab_size)
        if os.path.isfile(self.model_path):
            self.dpNet.load_weights(self.model_path)

    def predict(self, **data):
        '''
        使用模型
        :param path: 模型所在的路径
        :param name: 模型的名字
        :param data: 模型的输入参数
        :return:
        '''
        # latest = tf.train.latest_checkpoint(self.model_path)
        x_data = self.data.predict_data(**data)
        predict = self.dpNet.predict_classes(x_data)
        predict = self.data.to_categorys(predict)
        return predict

    def predict_all(self, datas):
        # latest = tf.train.latest_checkpoint(self.model_path)
        predicts = []
        for data in datas:
            x_data = self.data.predict_data(**data)
            predict = self.dpNet.predict_classes(x_data)
            predict = self.data.to_categorys(predict)
            predicts.append(predict)

        return predicts


if __name__ == '__main__':
    create_model(Processor().getWordsCount())
    print('xxx')
