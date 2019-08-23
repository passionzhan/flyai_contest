# -*- coding: utf-8 -*
import argparse
from functools import reduce

import numpy as np
from flyai.dataset import Dataset
import keras
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM
from keras_contrib.layers import CRF
# from keras_bert import extract_embeddings
from flyai.utils import remote_helper

from model import Model
from path import MODEL_PATH, LOG_PATH
import config
from utils import load_word2vec_embedding

# 超参
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=30, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=3, type=int, help="batch size")
args = parser.parse_args()
# 数据获取辅助类
dataset = Dataset(epochs=args.EPOCHS, batch=args.BATCH,val_batch=32)
# 模型操作辅助类
model = Model(dataset)

# 必须使用该方法下载模型，然后加载
path = remote_helper.get_remote_date("https://www.flyai.com/m/chinese_L-12_H-768_A-12.zip")
pre_trained_path = r'D:/jack_doc/python_src/flyai/chinese_L-12_H-768_A-12'

print("number of train examples:%d" % dataset.get_train_length())
print("number of validation examples:%d" % dataset.get_validation_length())

'''
keras: bi-LSTM+CRF
'''

# 得到训练和测试的数据
BiRNN_UNITS     = 2 * config.embeddings_size   # 双向RNN每步输出维数(2*单向维数)，  每个RNN(每个time step)输出维数， 设置成和 嵌入维数一样
EMBED_DIM       = config.embeddings_size      # 默认词向量的大小等于RNN(每个time step) 和 CNN(列) 中神经单元的个数, 为了避免混淆model中全部用unit_num表示。
TIME_STEP       = config.max_sequence      # 每个句子的最大长度和time_step一样,为了避免混淆model中全部用time_step表示。
DROPOUT_RATE    = config.dropout
LEARN_RATE      = config.leanrate
TAGS_NUM        = config.label_len
VOCAB_SIZE      = config.vocab_size + 2
LABEL_DIC       = config.label_dic

#
# texts = ['中 美 贸 易 战', '中 国 人 民 解 放 军 于 今 日 在 东 海 举 行 实 弹 演 习']
# embeddings = extract_embeddings(pre_trained_path, texts)

ner_model = Sequential()
embedding = Embedding(VOCAB_SIZE, EMBED_DIM, mask_zero=True,embeddings_initializer=load_word2vec_embedding(VOCAB_SIZE))
ner_model.add(embedding)
ner_model.add(Bidirectional(LSTM(BiRNN_UNITS // 2, return_sequences=True,dropout=DROPOUT_RATE)))
crf = CRF(len(LABEL_DIC), sparse_target=True)
ner_model.add(crf)
ner_model.summary()

ner_model.compile(optimizer=keras.optimizers.Adam(lr=LEARN_RATE), loss=crf.loss_function, metrics=[crf.accuracy])

max_val_acc, min_loss = 0, float('inf')
for i in range(dataset.get_step()):
    x_train, y_train, x_test, y_test = dataset.next_batch(args.BATCH)
    # padding
    x_train = np.asarray([list(x[:]) + (TIME_STEP - len(x)) * [config.src_padding] for x in x_train])
    y_train = np.asarray([list(y[:]) + (TIME_STEP - len(y)) * [TAGS_NUM - 1] for y in y_train])

    ner_model.train_on_batch(x_train,y_train)

    if i % 50 == 0 or i == dataset.get_step() - 1:

        val_acc = []
        val_loss = []
        for iLoop in range(6):
            # 此处获取的x_val样本数为dataset 的 val_batch == 6
            x_val, y_val = dataset.next_validation_batch()
            # padding
            x_val = np.asarray([list(x[:]) + (TIME_STEP - len(x)) * [config.src_padding] for x in x_val])
            y_val = np.asarray([list(y[:]) + (TIME_STEP - len(y)) * [TAGS_NUM - 1] for y in y_val])
            val_loss_and_metrics = ner_model.evaluate(x_val, y_val, verbose=0)
            val_loss.append(val_loss_and_metrics[0])
            val_acc.append(val_loss_and_metrics[1])

        cur_acc = reduce(lambda x, y: x + y, val_acc) / len(val_acc)
        cur_loss = reduce(lambda x, y: x + y, val_loss) / len(val_loss)

        print('step: %d/%d, val_loss: %f， val_acc: %f'
              % (i + 1, dataset.get_step(), cur_loss, cur_acc,))
        # val_loss_and_metrics[1],))

        if max_val_acc < cur_acc \
                or (max_val_acc == cur_acc and min_loss > cur_loss):
            max_val_acc, min_loss = cur_acc, cur_loss
            print('max_acc: %f, min_loss: %f' % (max_val_acc, min_loss))
            model.save_model(ner_model, overwrite=True)