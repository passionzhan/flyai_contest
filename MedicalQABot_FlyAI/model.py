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
    encode_state = [encode_h1, encode_h2, encode_c1, encode_c2]
    decode_input = Input(shape=(None,), dtype='int32', name='decode_input')
    decode_embedding_layer = Embedding(decode_vocab_size, output_dim=eDim, mask_zero=True, )
    decode_input_embedding = decode_embedding_layer(decode_input)
    decode_BiLSTM_layer = Bidirectional(LSTM(hide_dim, return_sequences=True, return_state=True, dropout=DROPOUT_RATE))
    decode_o, decode_h1, decode_h2, decode_c1, decode_c2 = \
        decode_BiLSTM_layer(decode_input_embedding, initial_state=encode_state)
    decode_dense_layer = Dense(decode_vocab_size, activation='softmax')
    decode_outputs = decode_dense_layer(decode_o)

    seq2seq_model = kModel([encode_input, decode_input], decode_outputs)

    seq2seq_model.compile(optimizer=Adam(lr=learning_rate, decay=1e-3),
                          loss='sparse_categorical_crossentropy',
                          metrics=[sparse_categorical_accuracy])

    encode_model = kModel(encode_input, encode_state)
    decode_state_input_h1 = Input(shape=(hide_dim,))
    decode_state_input_c1 = Input(shape=(hide_dim,))
    decode_state_input_h2 = Input(shape=(hide_dim,))
    decode_state_input_c2 = Input(shape=(hide_dim,))
    decode_states = [decode_state_input_h1, decode_state_input_h2, decode_state_input_c1, decode_state_input_c2]
    decode_outputs, decode_h1, decode_h2, decode_c1, decode_c2 = \
        decode_BiLSTM_layer(decode_embedding_layer(decode_input),initial_state=decode_states)
    decode_outputs = decode_dense_layer(decode_outputs)
    decode_model = kModel([decode_input] + decode_states,[decode_outputs,decode_h1, decode_h2, decode_c1, decode_c2])
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

    def decode_sequence(self, input_seq, max_decode_seq_length=max_ans_seq_len_predict,):
        # Encode the input as state vectors.
        states_value = self.encodeModel.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1))
        # Populate the first character of target sequence with the start character.
        # 将第一个词置为开始词
        target_seq[0, 0] = ans_dict['_sos_']
        target_seq_output = []
        output_len       = 0

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        while not stop_condition:
            output_tokens, h1, h2, c1, c2 = self.decodeModel.predict([target_seq] + states_value)

            # Sample a token
            sampled_word_index = np.argmax(output_tokens[0, -1, :])
            sampled_word = [sampled_word_index]

            # output_len += 1
            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_word == '_sos_' or
                    len(target_seq_output) > max_decode_seq_length):
                stop_condition = True

            # Add the sampled character to the sequence
            target_seq[0, 0] = sampled_word_index
            target_seq_output.append(sampled_word_index)
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


