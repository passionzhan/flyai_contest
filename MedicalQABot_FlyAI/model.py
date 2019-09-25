# -*- coding: utf-8 -*

import numpy as np
from flyai.model.base import Base
from keras.layers import Input, Embedding, LSTM, Dense, Bidirectional
from keras.models import Model as kModel
from keras.metrics import sparse_categorical_accuracy
from keras.optimizers import Adam

from path import MODEL_PATH, QA_MODEL_DIR
from data_helper import id2ans
from config import *

def create_model():
    encode_input = Input(shape=(None,), dtype='int32', name='encode_input')
    encode_input_embedding = Embedding(input_dim=encode_vocab_size, output_dim=eDim, mask_zero=True,)(encode_input)
    encode_BiLSTM_layer = Bidirectional(LSTM(hide_dim, return_sequences=False, return_state=True, dropout=DROPOUT_RATE),
                                        merge_mode=None,)
    encode_o1, encode_o2, encode_h1, encode_h2, encode_c1, encode_c2 = encode_BiLSTM_layer(encode_input_embedding)
    encode_state1 = [encode_h1, encode_c1]
    encode_state2 = [encode_h2, encode_c2]
    decode_input = Input(shape=(None,), dtype='int32', name='decode_input')
    decode_embedding_layer = Embedding(decode_vocab_size, output_dim=eDim, mask_zero=True, )
    decode_input_embedding = decode_embedding_layer(decode_input)
    decode_LSTM_layer1 = LSTM(hide_dim, return_sequences=True, return_state=True, dropout=DROPOUT_RATE)
    decode_LSTM_layer2 = LSTM(hide_dim, return_sequences=True, return_state=True, dropout=DROPOUT_RATE)
    decode_o1, decode_h1, decode_c1 = decode_LSTM_layer1(decode_input_embedding, initial_state=encode_state1)
    decode_o2, decode_h2, decode_c2 = decode_LSTM_layer2(decode_o1, initial_state=encode_state2)

    decode_dense_layer = Dense(decode_vocab_size, activation='softmax')
    decode_outputs = decode_dense_layer(decode_o2)

    seq2seq_model = kModel([encode_input, decode_input], decode_outputs)

    seq2seq_model.compile(optimizer=Adam(lr=learning_rate, decay=1e-3),
                          loss='sparse_categorical_crossentropy',
                          metrics=[sparse_categorical_accuracy])

    encode_model = kModel(encode_input, [encode_h1, encode_h2, encode_c1, encode_c2])
    decode_state_input_h1 = Input(shape=(hide_dim,))
    decode_state_input_c1 = Input(shape=(hide_dim,))
    decode_state_input_h2 = Input(shape=(hide_dim,))
    decode_state_input_c2 = Input(shape=(hide_dim,))
    decode_states1, decode_states2 = [decode_state_input_h1, decode_state_input_c1], [decode_state_input_h2, decode_state_input_c2]
    decode_outputs, decode_h1, decode_c1 = decode_LSTM_layer1(decode_embedding_layer(decode_input),initial_state=decode_states1)
    decode_outputs, decode_h2, decode_c2 = decode_LSTM_layer2(decode_outputs,initial_state=decode_states2)
    decode_outputs = decode_dense_layer(decode_outputs)
    decode_model = kModel([decode_input] + decode_states1 + decode_states2, [decode_outputs,decode_h1, decode_h2, decode_c1, decode_c2])
    return seq2seq_model, encode_model, decode_model

class Model(Base):
    def __init__(self, data):
        self.data = data
        self.model_path = os.path.join(MODEL_PATH, QA_MODEL_DIR)
        self.seq2seqModel, self.encodeModel, self.decodeModel = create_model()
        if os.path.isfile(self.model_path):
            print('加载训练好的模型：')
            self.seq2seqModel.load_weights(self.model_path)
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
            arg_topk = output_tokens.argsort(axis=-1)[:, -topk:]  # 每一项选出topk

            if output_len == 0:
                topk_sampled_word = arg_topk[0, :]
                target_seq_output = topk_sampled_word.reshap((topk,1))
                pre_score = []
                for idx in topk_sampled_word:
                    pre_score.append([output_tokens[0,idx]]*topk)
                pre_score = np.asarray(pre_score)
                target_seq  = topk_sampled_word.reshape((topk,1))
                h1, h2, c1, c2 = h1, h2, c1, c2
            else:
                cur_score = pre_score * np.sort(output_tokens, axis=-1)[:, -topk:],
                maxIdx = np.unravel_index(cur_score.argsort(axis=None)[-topk:],cur_score.shape)
                pre_score = np.tile(cur_score[maxIdx].reshape((topk,1)),(1,topk))
                target_seq = arg_topk[maxIdx].reshape((topk,1))
                target_seq_output = np.concatenate((target_seq_output, target_seq),axis=-1)
                h1, h2, c1, c2 = h1[maxIdx[0],:], h2[maxIdx[0],:], c1[maxIdx[0],:], c2[maxIdx[0],:]



            output_len += 1
            #
            # # Sample a token
            # sampled_word_index = np.argmax(output_tokens[0, -1, :])
            # sampled_word = [sampled_word_index]

            # output_len += 1
            # Exit condition: either hit max length
            # or find stop character.
            # if (sampled_word == '_sos_' or
            #         len(target_seq_output) > max_decode_seq_length):
            #     stop_condition = True
            #
            # # Add the sampled character to the sequence
            # target_seq[0, 0] = sampled_word_index
            # target_seq_output.append(sampled_word_index)
            # target_seq = np.concatenate([target_seq, word_idx], axis=1)

            # 状态更新
            states_value = [h1, h2, c1, c2]


        return [target_seq_output]

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


