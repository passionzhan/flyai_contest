# -*- coding: utf-8 -*-

import argparse

from keras.layers import Input, Embedding, LSTM, Dense, Bidirectional
from keras.models import Model
from keras.optimizers import Adam
from flyai.dataset import Dataset
from model import Model
from data_helper import *
from tensorflow.python.layers.core import Dense


'''
Tensorflow模版项目下载： https://www.flyai.com/python/tensorflow_template.zip
PyTorch模版项目下载： https://www.flyai.com/python/pytorch_template.zip
Keras模版项目下载： https://www.flyai.com/python/keras_template.zip
第一次使用请看项目中的：第一次使用请读我.html文件
常见问题请访问：https://www.flyai.com/question
意见和问题反馈有红包哦！添加客服微信：flyaixzs
'''

'''
项目中的超参
'''
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=10, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=64, type=int, help="batch size")
parser.add_argument("-vb", "--VAL_BATCH", default=64, type=int, help="val batch size")
args = parser.parse_args()
#  在本样例中， args.BATCH 和 args.VAL_BATCH 大小需要一致
'''
flyai库中的提供的数据处理方法
传入整个数据训练多少轮，每批次批大小
'''
dataset = Dataset(epochs=args.EPOCHS, batch=args.BATCH, val_batch=args.VAL_BATCH)
model = Model(dataset)

# region 词典构建
que_dict, ans_dict  = load_dict()
que_idx2word        = { v: k for k, v in que_dict.items()}
ans_idx2word        = { v: k for k, v in ans_dict.items()}
pad_idx = que_dict['_pad_']
word0   = que_idx2word[0]
que_dict['_pad_'], que_dict[word0]          = 0, pad_idx
que_idx2word[0], que_idx2word[pad_idx]      = '_pad_', word0
pad_idx = ans_dict['_pad_']
word0   = ans_idx2word[0]
ans_dict['_pad_'], ans_dict[word0]          = 0, pad_idx
ans_idx2word[0], ans_idx2word[pad_idx]      = '_pad_', word0

# # region 合并问题与答案词典
# for k, v in ans_dict.items():
#     que_dict.setdefault(k,len(que_dict))
# ans_dict = que_dict
# # endregion
# endregion

# 超参数
encode_vocab_size = len(que_dict)
decode_vocab_size = len(ans_dict)
# Batch Size,
batch_size = args.BATCH
# RNN Size
rnn_size = 64
# Number of Layers
num_layers = 3
# Embedding Size
encoding_embedding_size = 64
eDim = 200
hide_dim = 512
decoding_embedding_size = 64
# Learning Rate
learning_rate = 0.001
DROPOUT_RATE = 0.2

def create_model():
    encode_input = Input(shape=(None,), dtype='int32', name='encode_input')
    encode_input_embedding = Embedding(output_dim=eDim, mask_zero=True,)(encode_input)
    encode_BiLSTM_layer = Bidirectional(LSTM(hide_dim, return_sequences=False, return_state=True, dropout=DROPOUT_RATE))
    encode_outputs, encode_h, encode_c = encode_BiLSTM_layer(encode_input_embedding)
    encode_state = [encode_h, encode_c]
    decode_input = Input(shape=(None,), dtype='int32', name='decode_input')
    decode_input_embedding = Embedding(output_dim=eDim, mask_zero=True,)(decode_input)
    decode_BiLSTM_layer = Bidirectional(LSTM(hide_dim, return_sequences=True, dropout=DROPOUT_RATE))
    decode_outputs = decode_BiLSTM_layer(decode_input_embedding,initial_state=encode_state)
    decode_dense_layer = Dense(decoding_embedding_size, activation='softmax')
    decode_outputs = decode_dense_layer(decode_outputs)

    seq2seq_model = Model([encode_input, decode_input], decode_outputs)

    seq2seq_model.compile(optimizer=Adam(lr=learning_rate,decay=1e-3), loss='categorical_crossentropy')

    encode_model = Model(encode_input, encode_state)
    decode_state_input_h = Input(shape=(hide_dim,))
    decode_state_input_c = Input(shape=(hide_dim,))
    decode_states = [decode_state_input_h, decode_state_input_c]
    decode_outputs = decode_BiLSTM_layer(decode_input,
                                         initial_state=decode_states)
    decode_outputs = decode_dense_layer(decode_outputs)
    decode_model = Model([decode_input] + decode_states,decode_outputs)

    def decode_sequence(input_seq,max_decode_seq_length):
        # Encode the input as state vectors.
        states_value = encode_model.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1))
        # Populate the first character of target sequence with the start character.
        # 将第一个词置为开始词
        target_seq[0, 0] = ans_dict['_sos_']

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens = decode_model.predict([target_seq] + states_value)

            # Sample a token
            sampled_word_index = np.argmax(output_tokens[0, -1, :])
            sampled_word = ans_idx2word[sampled_word_index]
            decoded_sentence += sampled_word

            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_word == '_sos_' or
                    len(decoded_sentence) > max_decode_seq_length):
                stop_condition = True

            # Add the sampled character to the sequence
            word_idx = np.zeros((1, 1))
            word_idx[0, 0,] = sampled_word_index

            target_seq = np.concatenate([target_seq, word_idx], axis=1)

        return decoded_sentence

    # model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
    #           batch_size=batch_size,
    #           epochs=epochs,
    #           validation_split=0.2)
