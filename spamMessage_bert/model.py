# -*- coding: utf-8 -*
import os
import math

import numpy as np
from numpy import random
import tensorflow as tf
from flyai.model.base import Base

from bert import optimization
from path import *
from flyai.dataset import Dataset
from bert import modeling
from utilities import data_split


def create_model(is_training=True,):
    # region 模型超参数
    # is_training         = True
    # batch_size          = 256
    batch_size          = 256
    max_seq_len         = 256
    num_classes = 2

    #  创建bert的输入
    input_ids       = tf.placeholder(shape=[None, max_seq_len], dtype=tf.int32, name="input_ids")
    input_mask      = tf.placeholder(shape=[None, max_seq_len], dtype=tf.int32, name="input_mask")
    segment_ids     = tf.placeholder(shape=[None, max_seq_len], dtype=tf.int32, name="segment_ids")
    keep_prob       = tf.placeholder(tf.float32, name='keep_prob')
    # learning_rate   = tf.placeholder(tf.float32, name='learning_rate')
    learning_rate   = 0.01
    num_train_steps = tf.placeholder(tf.float32, name='num_train_steps')
    # num_train_steps = tf.placeholder(tf.int32, name='num_train_steps')
    ###
    input_labels = tf.placeholder(shape=[None,], dtype=tf.int32, name="input_labels")
    # 创建bert模型
    model = modeling.BertModel(
        config=BERT_CONFIG,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=False  # 这里如果使用TPU 设置为True，速度会快些。使用CPU 或GPU 设置为False ，速度会快些。
    )

    output_layer = model.get_pooled_output()  # 这个获取句子的output
    hidden_size = output_layer.shape[-1].value  # 获取输出的维度

    # 定义全连接层
    with tf.variable_scope("fc1"):
        output_layer = tf.nn.dropout(output_layer, keep_prob=keep_prob)

        fc1 = tf.get_variable(shape=[num_classes, hidden_size],dtype=tf.float32,initializer=tf.initializers.he_normal(),name="fc1")
        bias1 = tf.Variable(tf.zeros(shape=[num_classes,]),name='bias1')
        fc1 = tf.matmul(output_layer,fc1,transpose_b=True) + bias1

        # 分类器
        y_pred_cls = tf.argmax(tf.nn.softmax(fc1), 1, name='y_pred')  # 预测类别

    with tf.variable_scope("optimize"):
        # 将label进行onehot转化.
        one_hot_labels = tf.one_hot(input_labels, depth=2, dtype=tf.float32)
        # 损失函数，交叉熵
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=fc1, labels=one_hot_labels)
        loss = tf.reduce_mean(cross_entropy)
        # 优化器
        # train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        train_op = optimization.create_optimizer(loss, init_lr=learning_rate,
                                                 num_train_steps=num_train_steps,
                                                 num_warmup_steps = None,
                                                 use_tpu = False)

    with tf.variable_scope("accuracy"):
        correct_pred = tf.equal(input_labels, tf.cast(y_pred_cls,dtype=tf.int32))
        accuracy = tf.reduce_mean(tf.cast(correct_pred,dtype=tf.float32),name="accuracy")

    with tf.name_scope("summary"):
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("accuracy", accuracy)
        merged_summary = tf.summary.merge_all()

    # output_layer = model.get_sequence_output()  # 这个获取每个token的output 输入数据[batch_size, seq_length, embedding_size] 如果做seq2seq 或者ner 用这个

    # partial_init = tf.initializers.variables([filters, fc1, bias1, fc2, bias2], name='partial_init')

    inputParams = {'input_ids':input_ids,
                   'input_mask':input_mask,
                    'segment_ids':segment_ids,
                    'input_labels':input_labels,
                    'keep_prob':keep_prob,
                    # 'learning_rate':learning_rate,
                    'num_train_steps':num_train_steps
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

def gen_batch_data(x,y,batch_size):
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
        x_batch = x[bi:ei]
        y_batch = y[bi:ei]
        # x_batch, y_batch = x_data, y_data
        yield x_batch, y_batch

def conver2Input(x_batch, max_seq_len=256):
    seg_ids         = []
    mask_ids        = []
    input_ids       = []
    #  x 是np array
    for i, x in enumerate(x_batch):
        if len(x) > max_seq_len:
            seg_token = x[-1]
            x = x[0:max_seq_len]
            x[max_seq_len-1] = seg_token
            seg_token_idx = max_seq_len-1
        else:
            seg_token_idx = len(x) - 1
            x = np.concatenate((x, np.asarray([0] * (max_seq_len - len(x)))))

        input_ids.append(list(x))
        tmp_seg = [0] * max_seq_len
        tmp_seg[seg_token_idx] = 1
        tmp_mask = [0] * max_seq_len
        tmp_mask[seg_token_idx+1:] = [1] * (max_seq_len - seg_token_idx - 1)
        seg_ids.append(tmp_seg)
        mask_ids.append(tmp_mask)

    input_ids_batch = np.asarray(input_ids, dtype=np.int32)
    input_mask_batch = np.asarray(mask_ids, dtype=np.int32)
    segment_ids_batch = np.asarray(seg_ids, dtype=np.int32)
    return input_ids_batch, input_mask_batch, segment_ids_batch

class Model(Base):
    def __init__(self, data,):
        self.data = data
        # self.batch_size = 6
        self.model_path = os.path.join(MODEL_PATH, TENSORFLOW_MODEL_DIR)
        if os.path.exists(self.model_path):
            self.is_training = False
        else:
            self.is_training = True
        # self.vocab_size = Processor().getWordsCount()
        self.inputParams, self.outputParams, self.summaryParams = create_model(self.is_training)

    def train_model(self, epochs=32, val_ratio=0.1, train_batch_size=256,val_batch_size = None):
        # train_batch_size
        if val_batch_size is None:
            val_batch_size = 2 * train_batch_size
        x_train,y_train,x_val,y_val = data_split(self.data,val_ratio=val_ratio)
        train_len = x_train.shape[0]
        steps_per_epoch = math.ceil(train_len / train_batch_size)

        input_x         = self.inputParams['input_ids']
        input_x_mask    = self.inputParams['input_mask']
        input_x_seg     = self.inputParams['segment_ids']
        input_y         = self.inputParams['input_labels']
        keep_prob       = self.inputParams['keep_prob']
        # learning_rate   = self.inputParams['learning_rate']
        num_train_steps = self.inputParams['num_train_steps']

        loss = self.outputParams['loss']
        accuracy = self.outputParams['accuracy']
        train_op = self.outputParams['train_op']

        merged_summary = self.summaryParams['merged_summary']

        # 获取模型中所有的训练参数。
        tvars = tf.trainable_variables()
        # 加载BERT模型
        (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                                   BERT_CKPT)
        tf.train.init_from_checkpoint(BERT_CKPT, assignment_map)

        gen_train_batch = gen_batch_data(x_train, y_train, train_batch_size)
        gen_val_batch = gen_batch_data(x_val, y_val, val_batch_size)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            max_acc = 0
            noChangeEpoches = 0
            save_saver = tf.train.Saver()
            lr = 0.01
            for epoch in range(epochs):
                noChangedSteps = 0
                for j in range(steps_per_epoch):
                    x_train_batch, y_train_batch = next(gen_train_batch)
                    # processor_x   返回的是list 构成的np.arry
                    x_train_batch = self.data.processor_x(x_train_batch)
                    y_train_batch = self.data.processor_y(y_train_batch)

                    input_ids_batch, input_mask_batch, segment_ids_batch \
                        = conver2Input(x_train_batch,max_seq_len=256)

                    fetches = [loss, accuracy, train_op]

                    feed_dict = {
                        input_x:input_ids_batch,
                        input_x_mask:input_mask_batch,
                        input_x_seg:segment_ids_batch,
                        input_y:y_train_batch,
                        keep_prob:0.8,
                        # learning_rate:lr,
                        num_train_steps:epochs*steps_per_epoch,
                    }

                    loss_, accuracy_, _ = sess.run(fetches, feed_dict=feed_dict)
                    print('当前批次: {}/{}/{} | 当前训练损失: {} | 当前训练准确率： {}'
                          .format(j,steps_per_epoch, epoch, loss_, accuracy_,))

                    if j % 100 == 0 or j == self.data.get_step()-1:
                        x_val_batch, y_val_batch = next(gen_val_batch)
                        x_val_batch = self.data.processor_x(x_val_batch)
                        y_val_batch = self.data.processor_y(y_val_batch)
                        input_ids_val_batch, input_mask_val_batch, segment_ids_val_batch \
                            = conver2Input(x_val_batch, max_seq_len=256)

                        val_loss, val_acc = sess.run([loss, accuracy],
                                               feed_dict={
                                                    input_x:input_ids_val_batch,
                                                    input_x_mask:input_mask_val_batch,
                                                    input_x_seg:segment_ids_val_batch,
                                                    input_y:y_val_batch,
                                                    keep_prob:1,
                                                })
                        print('当前批次: {}/{}/{} | 当前验证集损失： {} | 当前验证集准确率： {}'
                              .format(j, steps_per_epoch, epoch, val_loss, val_acc,))
                        if val_acc > max_acc:
                            noChangedSteps  = 0
                            max_acc =  val_acc
                            if not os.path.exists(MODEL_PATH):
                                os.makedirs(MODEL_PATH)
                            save_path = save_saver.save(sess, self.model_path)
                            print("Model saved in path: %s" % save_path)
                        else:
                            noChangedSteps += 100

                # lr = 0.9*lr
                if noChangedSteps >= steps_per_epoch:
                    noChangeEpoches += 1
                else:
                    noChangeEpoches == 0

                if noChangeEpoches >= 3:
                    break


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
        keep_prob = self.inputParams['keep_prob']
        input_x         = self.inputParams['input_ids']
        input_x_mask    = self.inputParams['input_mask']
        input_x_seg     = self.inputParams['segment_ids']
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, self.model_path)
            for data in datas:
                x_data = self.data.predict_data(**data)
                input_ids_batch, input_mask_batch, segment_ids_batch = conver2Input(x_data, max_seq_len=256)
                predict = sess.run(y_pred_cls, feed_dict={input_x: input_ids_batch, input_x_mask:input_mask_batch,
                                   input_x_seg:segment_ids_batch,keep_prob: 1.0})
                predict = self.data.to_categorys(predict)
                predicts.append(predict)

        return predicts

if __name__ == '__main__':

    # inputParams, outputParams, summaryParams = create_model(Processor().getWordsCount())
    # # train_model(inputParams, outputParams, summaryParams,needInit=False)
    # dataset = Dataset(train_batch=128, val_batch=64, split_ratio = 0.9,)
    # model = Model(dataset)
    #
    # predic = model.predict(text="您好！我们这边是施华洛世奇鄞州万达店！您是我们尊贵的会员，特意邀请您参加我们x.x-x.x的三八女人节活动！满xxxx元享晶璨花漾丝巾")
    # print('xxx')
    #
    from bert import tokenization
    tokenizer = tokenization.FullTokenizer(VOCAB_FILE)
    tokens = tokenizer.tokenize("您好！我们这边是施华洛世奇鄞州万达店！您是我们尊贵的会员，特意邀请您参加我们x.x-x.x的三八女人节活动！满xxxx元享晶璨花漾丝巾")
    print(tokens)
