# -*- coding: utf-8 -*

import numpy as np
from flyai.model.base import Base
from keras.layers import Input, Embedding, LSTM, Dense, Bidirectional
from keras.models import Model as kModel
from keras.metrics import sparse_categorical_accuracy
from keras.optimizers import Adam

from bert4keras.bert import load_pretrained_model
from keras import backend as K
from bert4keras.utils import SimpleTokenizer, load_vocab

from path import MODEL_PATH, QA_MODEL_DIR
from data_helper import tokenizer,token_dict
from config import *

def create_model():
    bert4nlg_model = load_pretrained_model(
        BERT_CONFIG,
        BERT_CKPT,
        seq2seq=True,
        keep_words=None,

    )

    bert4nlg_model.summary()

    # 交叉熵作为loss，并mask掉输入部分的预测
    #  由于输入中，没有传入实际的 Y 值，所以各种准确度度量不起作用
    y_in = bert4nlg_model.input[0][:, 1:]  # 目标tokens
    y_mask = bert4nlg_model.input[1][:, 1:]
    y = bert4nlg_model.output[:, :-1]  # 预测tokens，预测与目标错开一位
    cross_entropy = K.sparse_categorical_crossentropy(y_in, y)
    cross_entropy = K.sum(cross_entropy * y_mask) / K.sum(y_mask)

    bert4nlg_model.add_loss(cross_entropy)
    bert4nlg_model.compile(optimizer=Adam(1e-5),)

    return bert4nlg_model

class Model(Base):
    def __init__(self, data):
        self.data = data
        self.model_path = os.path.join(MODEL_PATH, QA_MODEL_DIR)
        self.bert4nlg_model = create_model()
        if os.path.isfile(self.model_path):
            print('加载训练好的模型：')
            self.bert4nlg_model.load_weights(self.model_path)
            print('加载训练好的模型结束')

    # 基于beam reseach的解码
    def decode_sequence(self, input_seq, max_decode_seq_length=max_ans_seq_len_predict,topk=3):
        token_ids, segment_ids = tokenizer.encode(input_seq[:max_seq_len-2])
        # Encode the input as state vectors.
        input_seq = np.tile(token_ids,(topk,1))
        input_seg = np.tile(segment_ids,(topk,1))

        # target_seq  = np.array([[]] * topk,dtype=input_seq.dtype)
        # target_seq_output  = np.array([[]] * topk)
        pre_score   = np.array([[0.]*topk]*topk)

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        target_seq = None
        target_seq_output = None
        while not stop_condition:
            output_tokens = self.bert4nlg_model.predict([input_seq,input_seg])[:,-1,IGNORE_WORD_IDX:]

            # 当上次输出中有结束符 ‘[SEP]’ 时，将该样本输出'[SEP]'的概率置为最大1.0
            if target_seq is not None:
                for i, word in enumerate(target_seq[:,0]):
                    if word == token_dict['[SEP]']:
                        output_tokens[i,token_dict['[SEP]'] - IGNORE_WORD_IDX] = 1.0

            arg_topk = output_tokens.argsort(axis=-1)[:, -topk:]  # 每一项选出topk

            # pre_score
            # 取对数防止向下溢出
            # 利用对数计算，乘法改+法
            tmp_cur_score = np.log(np.sort(output_tokens, axis=-1)[:, -topk:])
            cur_score = pre_score + tmp_cur_score
            maxIdx = np.unravel_index(np.argsort(cur_score, axis=None)[-topk:],cur_score.shape)
            pre_score = np.tile(cur_score[maxIdx].reshape((topk,1)),(1,topk))
            target_seq  = arg_topk[maxIdx].reshape((topk,1)) + IGNORE_WORD_IDX
            if target_seq_output is None:
                target_seq_output = target_seq
            else:
                target_seq_output = np.concatenate((target_seq_output,target_seq),axis=-1)

            if (target_seq_output.shape[1] >= max_decode_seq_length
                    or (target_seq == token_dict['[SEP]'] * np.ones((topk,1))).all()):
                stop_condition = True

            targt_seg   = np.array([[1]] * topk,dtype=input_seg.dtype)
            input_seq = np.concatenate((input_seq,target_seq),axis=-1)
            input_seg = np.concatenate((input_seg,targt_seg),axis=-1)


        maxIdx = np.unravel_index(np.argmax(cur_score,axis=None), cur_score.shape)
        target_seq = arg_topk[maxIdx].reshape((1,))

        target_seq_output = np.concatenate((target_seq_output[maxIdx[0],:],target_seq),axis=-1).reshape(1,-1)
        for i, word in enumerate(target_seq_output[0,:]):
            if word == token_dict['[SEP]']:
                break
        target_seq_output = target_seq_output[:,0:i]
        return target_seq_output

    def predict(self, load_weights = False, **data):
        '''

        :param data:
        :return:
        '''

        x_data = data["que_text"]

        predict = self.decode_sequence(x_data,max_decode_seq_length=max_ans_seq_len_predict)
        predict = tokenizer.decode(predict[0])
        return predict

    def predict_all(self, datas):
        self.bert4nlg_modell.load_weights(self.model_path)
        predictions = []
        for data in datas:
            prediction = self.predict(**data)
            predictions.append(prediction)
        return predictions

