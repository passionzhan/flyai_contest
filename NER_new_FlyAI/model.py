#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : model.py
# @Author: Zhan
# @Date  : 8/26/2019
# @Desc  :

# -*- coding: utf-8 -*
import os

import keras
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Masking
from keras.initializers import constant
from crf import CRF
from crf_losses import crf_loss
from crf_accuracies import crf_accuracy
from keras.optimizers import Adam
import numpy as np
from flyai.model.base import Base

from path import MODEL_PATH
from utils import load_word2vec_embedding
import config



KERAS_MODEL_NAME = "my_BiLSTM_CRF.h5"

# 得到训练和测试的数据
BiRNN_UNITS     = 2 * config.embeddings_size   # 双向RNN每步输出维数(2*单向维数)，  每个RNN(每个time step)输出维数， 设置成和 嵌入维数一样
EMBED_DIM       = config.embeddings_size      # 默认词向量的大小等于RNN(每个time step) 和 CNN(列) 中神经单元的个数, 为了避免混淆model中全部用unit_num表示。
TIME_STEP       = config.max_sequence      # 每个句子的最大长度和time_step一样,为了避免混淆model中全部用time_step表示。
DROPOUT_RATE    = config.dropout
LEARN_RATE      = config.leanrate
TAGS_NUM        = config.label_len
VOCAB_SIZE      = config.vocab_size + 2
LABEL_DIC       = config.label_dic


class Model(Base):
    def create_NER_model(self):
        ner_model = Sequential()
        # keras_contrib 2.0.8, keras 2.2.5,下 当mmask_zero=True 会报
        # Tensors in list passed to 'values' of 'ConcatV2' Op have types [bool, float32] that don't all match.`
        # 错误。
        # 改成keras 2.2.4 解决
        embedding = Embedding(VOCAB_SIZE + 1, EMBED_DIM, mask_zero=True,
                              embeddings_initializer=constant(load_word2vec_embedding(config.vocab_size)))
        ner_model.add(embedding)
        # ner_model.add(Masking(mask_value=config.src_padding,))
        ner_model.add(Bidirectional(LSTM(BiRNN_UNITS // 2, return_sequences=True, dropout=DROPOUT_RATE)))
        crf = CRF(len(LABEL_DIC), sparse_target=True)
        ner_model.add(crf)
        # 以下两种损失和度量写法都可以
        ner_model.compile(Adam(lr=LEARN_RATE,decay=1e-3), loss=crf_loss, metrics=[crf_accuracy])
        # ner_model.compile(Adam(lr=LEARN_RATE), loss=crf.loss_function, metrics=[crf.accuracy])
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
    def predict(self,load_weights = False, **data):
        if load_weights:
            self.ner_model.load_weights(self.model_path)
        # 获取需要预测的图像数据， predict_data 方法默认会去调用 processor.py 中的 input_x 方法
        x_data = self.dataset.predict_data(**data)
        word_num = x_data.shape[1]
        x_data = np.asarray([list(x[:]) + (TIME_STEP - len(x)) * [config.src_padding] for x in x_data])

        predict = self.ner_model.predict(x_data)
        # 将预测数据转换成对应标签  to_categorys 会去调用 processor.py 中的 output_y 方法
        prediction = self.dataset.to_categorys(np.argmax(predict[0][:word_num],axis=-1))
        return prediction

    '''
    评估的时候会调用该方法实现评估得分
    '''

    def predict_all(self, datas):
        self.ner_model.load_weights(self.model_path)
        predictions = []
        for data in datas:
            prediction = self.predict(**data)
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

