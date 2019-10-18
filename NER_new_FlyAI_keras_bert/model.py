#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : model.py
# @Author: Zhan
# @Date  : 8/26/2019
# @Desc  :

# -*- coding: utf-8 -*
import os

import numpy as np
from flyai.model.base import Base
from keras.layers import Input, Lambda, Masking,Bidirectional, LSTM
from keras_bert import load_trained_model_from_checkpoint
from keras.optimizers import Adam
from keras import Model as kerasModel
import keras.backend as K
from crf import CRF
from crf_losses import crf_loss
from crf_accuracies import crf_accuracy

from processor import Processor
from path import *
import config


# 得到训练和测试的数据
BiRNN_UNITS     = 768   # BiLSTM 输出维数
EMBED_DIM       = config.embeddings_size      # 默认词向量的大小等于RNN(每个time step) 和 CNN(列) 中神经单元的个数, 为了避免混淆model中全部用unit_num表示。
TIME_STEP       = config.max_sequence      # 每个句子的最大长度和time_step一样,为了避免混淆model中全部用time_step表示。
DROPOUT_RATE    = config.dropout
LEARN_RATE      = config.leanrate
TAGS_NUM        = config.label_len
VOCAB_SIZE      = config.vocab_size + 2
LABEL_DIC       = config.label_dic

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
        tmp_mask = [1] * max_seq_len
        tmp_mask[seg_token_idx+1:] = [0] * (max_seq_len - seg_token_idx - 1)
        seg_ids.append(tmp_seg)
        mask_ids.append(tmp_mask)

    input_ids_batch = np.asarray(input_ids, dtype=np.int32)
    input_mask_batch = np.asarray(mask_ids, dtype=np.int32)
    segment_ids_batch = np.asarray(seg_ids, dtype=np.int32)
    return input_ids_batch, input_mask_batch, segment_ids_batch

class Model(Base):
    def create_NER_model(self):
        # 注意，尽管可以设置seq_len=None，但是仍要保证序列长度不超过512
        bert_model = load_trained_model_from_checkpoint(BERT_CONFIG, BERT_CKPT, seq_len=None)

        for layer in bert_model.layers:
            layer.trainable = True

        x1_in = Input(shape=(None,))
        x2_in = Input(shape=(None,))

        x = bert_model([x1_in, x2_in])
        x = Lambda(lambda x: x[:, 1:])(x)  # 取出每个单词对应的输出到CRF
        #  加入双向LSTM网络
        x = Bidirectional(LSTM(BiRNN_UNITS // 2, return_sequences=True, return_state=False, dropout=DROPOUT_RATE))(x)
        # x = K.concatenate(x)
        x = Masking(mask_value=0,)(x)
        rst = CRF(len(LABEL_DIC), sparse_target=True)(x)

        ner_model = kerasModel([x1_in, x2_in], rst)
        ner_model.compile(
            loss=crf_loss,
            metrics=[crf_accuracy],
            optimizer=Adam(LEARN_RATE),  # 用足够小的学习率
        )
        return ner_model

    def __init__(self, dataset):
        self.dataset = dataset
        self.model_path = os.path.join(MODEL_PATH, KERAS_MODEL_NAME)
        self.ner_model = self.create_NER_model()
        if os.path.isfile(self.model_path):
            print('加载训练好的模型：')
            self.ner_model.load_weights(self.model_path)
            print('加载训练好的模型结束')

    '''
    评估一条数据
    '''
    def predict(self, processor= None, load_weights = False, **data):
        if load_weights:
            self.ner_model.load_weights(self.model_path)
        # 获取需要预测的图像数据， predict_data 方法默认会去调用 processor.py 中的 input_x 方法
        x_data = self.dataset.predict_data(**data)
        word_num = x_data[0].shape[0] - 1 #  去掉首字符 '[cls]'
        x_batch_ids, x_batch_mask, x_batch_seg = conver2Input(x_data, max_seq_len=config.max_sequence)
        #
        # word_num = x_data.shape[1]
        # word_num
        # x_data = np.asarray([list(x_data[0])])

        predict = self.ner_model.predict([x_batch_ids, x_batch_mask,])
        # 将预测数据转换成对应标签  to_categorys 会去调用 processor.py 中的 output_y 方法
        # word_num - 1 去掉结尾的 '[sep]'字符。
        prediction = self.dataset.to_categorys(np.argmax(predict[0][:word_num-1],axis=-1))
        if processor is None:
            processor = Processor()
        prediction = processor.processedOutput(data['source'],x_data[0],prediction)
        return prediction

    '''
    评估的时候会调用该方法实现评估得分
    '''

    def predict_all(self, datas):
        self.ner_model.load_weights(self.model_path)
        predictions = []
        processor = Processor()
        for data in datas:
            prediction = self.predict(processor=processor,**data)
            predictions.append(prediction)
        return predictions

    '''
    保存模型的方法
    '''
    def save_model(self, model, overwrite=False):
        super().save_model(model, MODEL_PATH, KERAS_MODEL_NAME, overwrite)
        ### 加入模型保存代码
        if not os.path.exists(MODEL_PATH):
            os.makedirs(MODEL_PATH)
        model.save_weights(self.model_path)
        # save_model(model,self.model_path, overwrite, include_optimizer=False)

