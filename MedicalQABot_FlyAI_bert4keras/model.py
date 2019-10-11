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
from data_helper import id2ans
from config import *

def create_model():
    bert4nlg_model = load_pretrained_model(
        BERT_CONFIG,
        BERT_CKPT,
        seq2seq=True,
        keep_words=None
    )

    bert4nlg_model.summary()

    # 交叉熵作为loss，并mask掉输入部分的预测
    y_in = bert4nlg_model.input[0][:, 1:]  # 目标tokens
    y_mask = bert4nlg_model.input[1][:, 1:]
    y = bert4nlg_model.output[:, :-1]  # 预测tokens，预测与目标错开一位
    cross_entropy = K.sparse_categorical_crossentropy(y_in, y)
    cross_entropy = K.sum(cross_entropy * y_mask) / K.sum(y_mask)

    bert4nlg_model.add_loss(cross_entropy)
    bert4nlg_model.compile(optimizer=Adam(1e-5))

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
        # Encode the input as state vectors.
        input_seq = np.tile(input_seq,(topk,1))
        states_value = self.encodeModel.predict(input_seq)

        # Generate empty target sequence of length 1.
        # 将第一个词置为开始词
        target_seq = np.array([[ans_dict['_sos_']]] * topk)
        # Populate the first character of target sequence with the start character.
        target_seq_output = []
        output_len       = 0
        pre_score = np.array([[1.]*topk]*topk)

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        while not stop_condition:
            output_tokens, h1, h2, c1, c2 = self.decodeModel.predict([target_seq] + states_value)
            output_tokens = output_tokens.reshape((topk,-1))

            # 当上次输出中有结束符 ‘_eos_’ 时，将该样本输出'_sos_'的概率置为最大1.0
            for i, word in enumerate(target_seq[:,0]):
                if word == ans_dict['_eos_']:
                    output_tokens[i,ans_dict['_eos_']] = 1.0

            arg_topk = output_tokens.argsort(axis=-1)[:, -topk:]  # 每一项选出topk

            if output_len == 0:
                topk_sampled_word = arg_topk[0, :]
                target_seq_output = topk_sampled_word.reshape((topk,1))
                pre_score = []
                for idx in topk_sampled_word:
                    pre_score.append([output_tokens[0,idx]]*topk)
                pre_score = np.asarray(pre_score)
                # 取对数防止向下溢出
                pre_score = np.log(pre_score)
                target_seq  = topk_sampled_word.reshape((topk,1))
                h1, h2, c1, c2 = h1, h2, c1, c2
            else:
                # pre_score
                # 取对数防止向下溢出
                # 利用对数计算，乘法该+法
                tmp_cur_score = np.log(np.sort(output_tokens, axis=-1)[:, -topk:])
                cur_score = pre_score + tmp_cur_score
                # cur_score = np.log(cur_score)
                maxIdx = np.unravel_index(np.argsort(cur_score, axis=None)[-topk:],cur_score.shape)
                pre_score = np.tile(cur_score[maxIdx].reshape((topk,1)),(1,topk))
                target_seq = arg_topk[maxIdx].reshape((topk,1))
                target_seq_output = np.concatenate((target_seq_output, target_seq),axis=-1)
                h1, h2, c1, c2 = h1[maxIdx[0],:], h2[maxIdx[0],:], c1[maxIdx[0],:], c2[maxIdx[0],:]

            output_len += 1
            # sampled_word == '_sos_' or
            # Exit condition: either hit max length
            # or find stop character.
            if (target_seq_output.shape[1] >= max_decode_seq_length
                    or (target_seq == ans_dict['_eos_'] * np.ones((topk,1))).all()):
                stop_condition = True
            #
            # # Add the sampled character to the sequence
            # target_seq[0, 0] = sampled_word_index
            # target_seq_output.append(sampled_word_index)
            # target_seq = np.concatenate([target_seq, word_idx], axis=1)

            # 状态更新
            states_value = [h1, h2, c1, c2]

        maxIdx = np.unravel_index(np.argmax(cur_score,axis=None), cur_score.shape)
        target_seq = arg_topk[maxIdx].reshape((1,))

        target_seq_output = np.concatenate((target_seq_output[maxIdx[0],:],target_seq),axis=-1).reshape(1,-1)
        for i, word in enumerate(target_seq_output[0,:]):
            if word == ans_dict['_eos_']:
                break
        target_seq_output = target_seq_output[:,0:i]
        return target_seq_output

    def predict(self, load_weights = False, **data):
        '''

        :param data:
        :return:
        '''
        x_data = self.data.predict_data(**data)
        que_x, que_len = x_data

        predict = self.decode_sequence(que_x,max_decode_seq_length=max_ans_seq_len_predict)

        return self.data.to_categorys(predict)

    def predict_all(self, datas):
        self.seq2seqModel.load_weights(self.model_path)
        predictions = []
        for data in datas:
            prediction = self.predict(**data)
            predictions.append(prediction)
        return predictions


