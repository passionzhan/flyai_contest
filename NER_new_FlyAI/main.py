# -*- coding: utf-8 -*
import argparse
from functools import reduce

import numpy as np
from flyai.dataset import Dataset
import keras
from flyai.utils import remote_helper

from model import Model
import config

# 超参
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=2, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=5, type=int, help="batch size")
args = parser.parse_args()
# 数据获取辅助类
dataset = Dataset(epochs=args.EPOCHS, batch=args.BATCH,val_batch=args.BATCH)
# 模型操作辅助类
model = Model(dataset)

# 必须使用该方法下载模型，然后加载
# path = remote_helper.get_remote_date("https://www.flyai.com/m/chinese_L-12_H-768_A-12.zip")
# pre_trained_path = r'D:/jack_doc/python_src/flyai/chinese_L-12_H-768_A-12'

print("number of train examples:%d" % dataset.get_train_length())
print("number of validation examples:%d" % dataset.get_validation_length())

'''
keras: bi-LSTM+CRF
'''

# 得到训练和测试的数据
TIME_STEP       = config.max_sequence      # 每个句子的最大长度和time_step一样,为了避免混淆model中全部用time_step表示。
TAGS_NUM        = config.label_len

#
# texts = ['中 美 贸 易 战', '中 国 人 民 解 放 军 于 今 日 在 东 海 举 行 实 弹 演 习']
# embeddings = extract_embeddings(pre_trained_path, texts)

ner_model = model.ner_model
ner_model.summary()

max_val_acc, min_loss = 0, float('inf')
for i in range(dataset.get_step()):
    x_train, y_train = dataset.next_train_batch()
    # padding
    x_train = np.asarray([list(x[:]) + (TIME_STEP - len(x)) * [config.src_padding] for x in x_train])
    y_train = np.asarray([list(y[:]) + (TIME_STEP - len(y)) * [TAGS_NUM - 1] for y in y_train])
    y_train = y_train.reshape(y_train.shape[0],y_train.shape[1],1)
    ner_model.train_on_batch(x_train,y_train)

    if i % 50 == 0 or i == dataset.get_step() - 1:

        x_val, y_val = dataset.next_validation_batch()
        # padding
        x_val = np.asarray([list(x[:]) + (TIME_STEP - len(x)) * [config.src_padding] for x in x_val])
        y_val = np.asarray([list(y[:]) + (TIME_STEP - len(y)) * [TAGS_NUM - 1] for y in y_val])
        y_val = y_train.reshape(y_val.shape[0], y_val.shape[1], 1)
        val_loss_and_metrics = ner_model.evaluate(x_val, y_val, verbose=0)
        cur_loss = val_loss_and_metrics[0]
        cur_acc = val_loss_and_metrics[1]


        print('step: %d/%d, val_loss: %f， val_acc: %f'
              % (i + 1, dataset.get_step(), cur_loss, cur_acc,))
        # val_loss_and_metrics[1],))

        if max_val_acc < cur_acc \
                or (max_val_acc == cur_acc and min_loss > cur_loss):
            max_val_acc, min_loss = cur_acc, cur_loss
            print('max_acc: %f, min_loss: %f' % (max_val_acc, min_loss))
            model.save_model(ner_model, overwrite=True)