# -*- coding: utf-8 -*-

import argparse

import tensorflow as tf
from keras.layers import Input, Embedding, LSTM, Dense, Bidirectional
from keras.models import Model
from keras.optimizers import Adam
from flyai.dataset import Dataset
from model import Model
from path import MODEL_PATH, LOG_PATH
from data_helper import *
from tensorflow.python.layers.core import Dense
from flyai.utils.log_helper import train_log



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

# 超参数
que_dict, ans_dict = load_dict()

# region 合并问题与答案词典
# for k, v in ans_dict.items():
#     que_dict.setdefault(k,len(que_dict))
# ans_dict = que_dict
# endregion

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
    encode_input = Input(shape=(None, encode_vocab_size), dtype='int32', name='encode_input')
    encode_input_embedding = Embedding(output_dim=eDim, mask_zero=True,)(encode_input)
    encode_BiLSTM_layer = Bidirectional(LSTM(hide_dim, return_sequences=False, return_state=True, dropout=DROPOUT_RATE))
    encode_outputs, encode_h, encode_c = encode_BiLSTM_layer(encode_input_embedding)
    encode_state = [encode_h, encode_c]
    decode_input = Input(shape=(None, decoding_embedding_size), dtype='int32', name='decode_input')
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

    def decode_sequence(input_seq):
        # Encode the input as state vectors.
        states_value = encode_model.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, decode_vocab_size))
        # Populate the first character of target sequence with the start character.
        # 将第一个词置为开始词
        target_seq[0, 0, target_token_index['\t']] = 1.

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens = decode_model.predict([target_seq] + states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = reverse_target_char_index[sampled_token_index]
            decoded_sentence += sampled_char

            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_char == '\n' or
                    len(decoded_sentence) > max_decoder_seq_length):
                stop_condition = True

            # Add the sampled character to the sequence
            char_vector = np.zeros((1, 1, num_decoder_tokens))
            char_vector[0, 0, sampled_token_index] = 1.

            target_seq = np.concatenate([target_seq, char_vector], axis=1)

        return decoded_sentence
    # model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
    #           batch_size=batch_size,
    #           epochs=epochs,
    #           validation_split=0.2)

# 输入层
def get_inputs():
    inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    # 定义target序列最大长度（之后target_sequence_length和source_sequence_length会作为feed_dict的参数）
    target_sequence_length = tf.placeholder(tf.int32, (None,), name='target_sequence_length')
    max_target_sequence_length = tf.reduce_max(target_sequence_length, name='max_target_len')
    source_sequence_length = tf.placeholder(tf.int32, (None,), name='source_sequence_length')

    return inputs, targets, learning_rate, target_sequence_length, max_target_sequence_length, source_sequence_length


# Encoder
"""
在Encoder端，我们需要进行两步，第一步要对我们的输入进行Embedding，再把Embedding以后的向量传给RNN进行处理。
在Embedding中，我们使用tf.contrib.layers.embed_sequence，它会对每个batch执行embedding操作。
"""


def get_encoder_layer(input_data, rnn_size, num_layers, source_sequence_length, source_vocab_size,
                      encoding_embedding_size):
    """
    构造Encoder层
    参数说明：
    - input_data: 输入tensor
    - rnn_size: rnn隐层结点数量
    - num_layers: 堆叠的rnn cell数量
    - source_sequence_length: 源数据的序列长度
    - source_vocab_size: 源数据的词典大小
    - encoding_embedding_size: embedding的大小
    """
    encoder_embed_input = tf.contrib.layers.embed_sequence(input_data, source_vocab_size, encoding_embedding_size)

    def get_lstm_cell(rnn_size):
        lstm_cell = tf.nn.rnn_cell.LSTMCell(rnn_size,initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        # lstm_cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        return lstm_cell

    cell = tf.contrib.rnn.MultiRNNCell([get_lstm_cell(rnn_size) for _ in range(num_layers)])

    encoder_output, encoder_state = \
        tf.nn.dynamic_rnn(cell, encoder_embed_input,
                                    sequence_length=source_sequence_length, dtype=tf.float32)

    return encoder_output, encoder_state


def process_decoder_input(data, phonem_dict, batch_size):
    ending = tf.strided_slice(data, [0, 0], [batch_size, -1], [1, 1])
    decoder_input = tf.concat([tf.fill([batch_size, 1], phonem_dict['_sos_']), ending], 1)

    return decoder_input


def decoding_layer(phonem_dict, decoding_embedding_size, num_layers, rnn_size,
                   target_sequence_length, max_target_sequence_length, encoder_state, decoder_input):
    '''
    构造Decoder层
    参数：
    - target_letter_to_int: target数据的映射表
    - decoding_embedding_size: embed向量大小
    - num_layers: 堆叠的RNN单元数量
    - rnn_size: RNN单元的隐层结点数量
    - target_sequence_length: target数据序列长度
    - max_target_sequence_length: target数据序列最大长度
    - encoder_state: encoder端编码的状态向量
    - decoder_input: decoder端输入
    '''

    # 1. Embedding
    target_vocab_size = len(phonem_dict)
    decoder_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoding_embedding_size]))
    decoder_embed_input = tf.nn.embedding_lookup(decoder_embeddings, decoder_input)

    # 构造Decoder中的RNN单元
    def get_decoder_cell(rnn_size):
        decoder_cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        return decoder_cell

    cell = tf.contrib.rnn.MultiRNNCell([get_decoder_cell(rnn_size) for _ in range(num_layers)])

    # Output全连接层
    # target_vocab_size定义了输出层的大小
    output_layer = Dense(target_vocab_size, kernel_initializer=tf.truncated_normal_initializer(mean=0.1, stddev=0.1))

    # 4. Training decoder
    with tf.variable_scope("decode"):
        training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_embed_input,
                                                            sequence_length=target_sequence_length,
                                                            time_major=False)

        training_decoder = tf.contrib.seq2seq.BasicDecoder(cell, training_helper, encoder_state, output_layer)
        training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder, impute_finished=True,
                                                                          maximum_iterations=max_target_sequence_length)

    # 5. Predicting decoder
    # 与training共享参数

    with tf.variable_scope("decode", reuse=True):
        # 创建一个常量tensor并复制为batch_size的大小
        start_tokens = tf.tile(tf.constant([phonem_dict['_sos_']], dtype=tf.int32),
                               [tf.shape(target_sequence_length)[0]], name='start_token')
        predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_embeddings, start_tokens,
                                                                     phonem_dict['_eos_'])

        predicting_decoder = tf.contrib.seq2seq.BasicDecoder(cell,
                                                             predicting_helper,
                                                             encoder_state,
                                                             output_layer)
        predicting_decoder_output, _, _ = \
            tf.contrib.seq2seq.dynamic_decode(predicting_decoder,
                                              impute_finished=True, maximum_iterations=max_target_sequence_length)

    return training_decoder_output, predicting_decoder_output


# 上面已经构建完成Encoder和Decoder，下面将这两部分连接起来，构建seq2seq模型
def seq2seq_model(input_data, targets, target_sequence_length, max_target_sequence_length,
                  source_sequence_length, source_vocab_size, rnn_size, num_layers):
    _, encoder_state = get_encoder_layer(input_data,
                                         rnn_size,
                                         num_layers,
                                         source_sequence_length,
                                         source_vocab_size,
                                         encoding_embedding_size)

    decoder_input = process_decoder_input(targets, ans_dict, batch_size)

    training_decoder_output, predicting_decoder_output = decoding_layer(ans_dict,
                                                                        decoding_embedding_size,
                                                                        num_layers,
                                                                        rnn_size,
                                                                        target_sequence_length,
                                                                        max_target_sequence_length,
                                                                        encoder_state,
                                                                        decoder_input)

    return training_decoder_output, predicting_decoder_output


# 构造graph
train_graph = tf.Graph()
with train_graph.as_default():
    global_step = tf.Variable(0, name="global_step", trainable=False)
    input_data, targets, lr, target_sequence_length, max_target_sequence_length, source_sequence_length = get_inputs()

    training_decoder_output, predicting_decoder_output = seq2seq_model(input_data,
                                                                       targets,
                                                                       target_sequence_length,
                                                                       max_target_sequence_length,
                                                                       source_sequence_length,
                                                                       len(que_dict),
                                                                       rnn_size,
                                                                       num_layers)

    training_logits = tf.identity(training_decoder_output.rnn_output, 'logits')
    predicting_logits = tf.identity(predicting_decoder_output.sample_id, name='predictions')
    masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name="masks")

    # phone_accuracy
    logits_flat = tf.reshape(training_logits, [-1, decode_vocab_size])
    predict = tf.cast(tf.reshape(tf.argmax(logits_flat, 1), [tf.shape(input_data)[0], -1]),
                      tf.int32, name='predict')
    corr_target_id_cnt = tf.cast(tf.reduce_sum(
        tf.cast(tf.equal(tf.cast(targets, tf.float32), tf.cast(predict, tf.float32)),
                tf.float32) * masks), tf.int32)
    ans_accuracy = corr_target_id_cnt / tf.reduce_sum(target_sequence_length)

    with tf.name_scope("optimization"):
        cost = tf.contrib.seq2seq.sequence_loss(training_logits, targets, masks)
        optimizer = tf.train.AdamOptimizer(lr)

        # 对var_list中的变量计算loss的梯度 该函数为函数minimize()的第一部分，返回一个以元组(gradient, variable)组成的列表
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
        # 将计算出的梯度应用到变量上，是函数minimize()的第二部分，返回一个应用指定的梯度的操作Operation，对global_step做自增操作
        train_op = optimizer.apply_gradients(capped_gradients)
        summary_op = tf.summary.merge([tf.summary.scalar("loss", cost)])


with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter(LOG_PATH, sess.graph)

    for step in range(dataset.get_step()):
        que_train, ans_train = dataset.next_train_batch()
        que_val, ans_val = dataset.next_validation_batch()

        que_x, que_length = que_train
        ans_x, ans_lenth = ans_train
        ans_x = process_ans_batch(ans_x, ans_dict, int(sorted(list(ans_lenth), reverse=True)[0]))

        feed_dict = {input_data: que_x,
                     targets: ans_x,
                     lr: learning_rate,
                     target_sequence_length: ans_lenth,
                     source_sequence_length: que_length}
        fetches = [train_op, cost, training_logits, ans_accuracy]
        _, tra_loss, logits, train_acc = sess.run(fetches, feed_dict=feed_dict)

        val_que_x, val_que_len = que_val
        val_ans_x, val_ans_len = ans_val
        val_ans_x = process_ans_batch(val_ans_x, ans_dict, int(sorted(list(val_ans_len), reverse=True)[0]))
        feed_dict = {input_data: val_que_x,
                     targets: val_ans_x,
                     lr: learning_rate,
                     target_sequence_length: val_ans_len,
                     source_sequence_length: val_que_len}

        val_loss, val_acc = sess.run([cost, ans_accuracy], feed_dict=feed_dict)

        summary = sess.run(summary_op, feed_dict=feed_dict)
        train_writer.add_summary(summary, step)

        # 调用系统打印日志函数，这样在线上可看到训练和校验准确率和损失的实时变化曲线
        train_log(train_loss=tra_loss, train_acc=train_acc, val_loss=val_loss, val_acc=val_acc)

        # 实现自己的保存模型逻辑
        if step % 200 == 0:
            model.save_model(sess, MODEL_PATH, overwrite=True)
    model.save_model(sess, MODEL_PATH, overwrite=True)
