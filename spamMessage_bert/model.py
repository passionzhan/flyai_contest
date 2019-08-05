# -*- coding: utf-8 -*
import os
import math

import tensorflow as tf
from flyai.model.base import Base

from path import MODEL_PATH, LOG_PATH
from processor import Processor
from flyai.dataset import Dataset
from flyai.utils import remote_helper

TENSORFLOW_MODEL_DIR = "dpNet.ckpt"

def create_model(vocab_size, ):
    # region 模型超参数
    e_dim = 768
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
        bias1 = tf.Variable(tf.zeros(shape=[fc1_dim,]),name='bias1')
        fc1_mid = tf.matmul(pool,fc1,transpose_b=True) + bias1
        fc1_drop = tf.nn.dropout(fc1_mid,keep_prob=keep_prob)
        fc1_rst =  tf.nn.relu(fc1_drop)

    # 定义全连接层2 + 分类层
    with tf.variable_scope("fc2"):
        fc2 = tf.get_variable(shape=[2,fc1_dim],dtype=tf.float32,initializer=tf.initializers.he_normal(),name="fc2")
        bias2 = tf.Variable(tf.zeros(shape=[2,]),name='bias2')
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

    partial_init = tf.initializers.variables([filters, fc1, bias1, fc2, bias2], name='partial_init')

    inputParams = {'input_x':input_x,
                    'input_y':input_y,
                    'keep_prob':keep_prob,
                   }

    outputParams ={'loss':loss,
                   'y_pred_cls':y_pred_cls,
                   'accuracy':accuracy,
                   'train_op':train_op,
    }

    summaryParams = {
        'merged_summary':merged_summary
    }

    return inputParams,outputParams,summaryParams

class Model(Base):
    def __init__(self, data,):
        self.data = data
        self.model_path = os.path.join(MODEL_PATH, TENSORFLOW_MODEL_DIR)
        self.vocab_size = Processor().getWordsCount()
        self.inputParams, self.outputParams, self.summaryParams = create_model(self.vocab_size)

    def train_model(self, needInit=True, loadmodelType= 1, epochs=2, ):
        input_x = self.inputParams['input_x']
        input_y = self.inputParams['input_y']
        keep_prob = self.inputParams['keep_prob']

        loss = self.outputParams['loss']
        accuracy = self.outputParams['accuracy']
        train_op = self.outputParams['train_op']

        merged_summary = self.summaryParams['merged_summary']
        with tf.Session() as sess:
            default_graph = sess.graph
            if needInit:
                init = tf.global_variables_initializer()
                sess.run(init)
            elif loadmodelType == 'all': # all表示加载所有变量
                init_saver = tf.train.Saver()
                init_saver.restore(sess, os.path.join(MODEL_PATH, TENSORFLOW_MODEL_DIR))
            else: #  其他表示紧加载词嵌入
                init_saver = tf.train.Saver({"bert/embeddings/word_embeddings":
                                                 default_graph.get_tensor_by_name("embedding/encoder_embedding:0")})
                # 必须使用该方法下载模型，然后加载
                path = remote_helper.get_remote_date("https://www.flyai.com/m/chinese_L-12_H-768_A-12.zip")
                init_saver.restore(sess, path)
                # 本地测试用
                # init_saver.restore(sess, os.path.join(os.getcwd(),'chinese_L-12_H-768_A-12','bert_model.ckpt'))


                global_vars = tf.global_variables()
                is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
                not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
                # print([str(i.name) for i in not_initialized_vars])  # only for testing
                if len(not_initialized_vars):
                    sess.run(tf.variables_initializer(not_initialized_vars))

                filters = default_graph.get_tensor_by_name("cnn/filters:0")
                fc1     = default_graph.get_tensor_by_name('fc1/fc1:0')
                bias1   = default_graph.get_tensor_by_name('fc1/bias1:0')
                fc2     = default_graph.get_tensor_by_name('fc2/fc2:0')
                bias2   = default_graph.get_tensor_by_name('fc2/bias2:0')

                # partial_init = default_graph.get_operation_by_name('partial_init')
                # default_graph.g
                # sess.run(partial_init)


            train_writer = tf.summary.FileWriter(LOG_PATH, sess.graph)

            # dataset = Dataset(train_batch=128, val_batch=64, split_ratio = 0.9,)
            # epochs = 2

            # step = math.ceil(self.data.get_train_length() / min(256,))
            max_acc = 0
            min_loss = 0
            save_saver = tf.train.Saver()
            for j in range(self.data.get_step()):
                x_train, y_train = self.data.next_train_batch()

                fetches = [loss, accuracy, train_op]

                feed_dict = {input_x: x_train, input_y: y_train, keep_prob: 0.8}
                loss_, accuracy_, _ = sess.run(fetches, feed_dict=feed_dict)

                if j % 100 == 0 or j == self.data.get_step()-1:
                    summary_train = sess.run(merged_summary, feed_dict=feed_dict, )
                    train_writer.add_summary(summary_train, j)

                    nSmp_val = 0
                    nCount = 0
                    ave_loss = 0
                    for i in range(10):
                        x_val, y_val = self.data.next_validation_batch()
                        summary_val = sess.run([loss, accuracy],
                                               feed_dict={input_x: x_val, input_y: y_val, keep_prob: 1.0})
                        nSmp_val += x_val.shape[0]
                        nCount += summary_val[1] * x_val.shape[0]
                        ave_loss += summary_val[0]
                    val_accuracy = nCount / nSmp_val
                    ave_loss = ave_loss / 10
                    print('当前批次: {} | 当前训练损失: {} | 当前训练准确率： {} | '
                              '当前验证集损失： {} | 当前验证集准确率： {}'.format(j, loss_, accuracy_, ave_loss,
                                                                  val_accuracy))
                    if val_accuracy > max_acc or (val_accuracy == max_acc and ave_loss < min_loss):
                        max_acc, min_loss = val_accuracy, ave_loss
                        save_path = save_saver.save(sess, os.path.join(MODEL_PATH, TENSORFLOW_MODEL_DIR))
                        print("Model saved in path: %s" % save_path)

    def model_predict(self, x_data):
        y_pred_cls = self.outputParams['y_pred_cls']
        input_x = self.inputParams['input_x']
        keep_prob = self.inputParams['keep_prob']
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, os.path.join(MODEL_PATH, TENSORFLOW_MODEL_DIR))
            predict = sess.run(y_pred_cls, feed_dict={input_x: x_data, keep_prob: 1.0})

        return predict

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
        predict = self.model_predict(x_data)
        predict = self.data.to_categorys(predict)
        return predict

    def predict_all(self, datas):
        predicts = []
        y_pred_cls = self.outputParams['y_pred_cls']
        input_x = self.inputParams['input_x']
        keep_prob = self.inputParams['keep_prob']
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, os.path.join(MODEL_PATH, TENSORFLOW_MODEL_DIR))
            for data in datas:
                x_data = self.data.predict_data(**data)
                predict = sess.run(y_pred_cls, feed_dict={input_x: x_data, keep_prob: 1.0})
                predict = self.data.to_categorys(predict)
                predicts.append(predict)

        return predicts

if __name__ == '__main__':

    # inputParams, outputParams, summaryParams = create_model(Processor().getWordsCount())
    # train_model(inputParams, outputParams, summaryParams,needInit=False)
    dataset = Dataset(train_batch=128, val_batch=64, split_ratio = 0.9,)
    model = Model(dataset)

    predic = model.predict(text="您好！我们这边是施华洛世奇鄞州万达店！您是我们尊贵的会员，特意邀请您参加我们x.x-x.x的三八女人节活动！满xxxx元享晶璨花漾丝巾")
    print('xxx')
