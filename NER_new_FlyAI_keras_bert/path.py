# -*- coding: utf-8 -*
import sys
import os

from flyai.utils import remote_helper

# 训练数据的路径
DATA_PATH = os.path.join(sys.path[0], 'data', 'input')
# 模型保存的路径
MODEL_PATH = os.path.join(sys.path[0], 'data', 'output', 'model')
# 训练log的输出路径
LOG_PATH = os.path.join(sys.path[0], 'data', 'output', 'logs')

# 必须使用该方法下载模型，然后加载
# BERT_PATH = remote_helper.get_remote_date("https://www.flyai.com/m/chinese_L-12_H-768_A-12.zip")
# print('BERT_PATH:{}'.format(BERT_PATH))
# BERT_PATH = os.path.dirname(BERT_PATH)
# BERT_PATH = os.path.join(BERT_PATH, 'chinese_L-12_H-768_A-12')
# BERT_PATH = r'D:\Study\flyai_contest\chinese_L-12_H-768_A-12'
BERT_PATH = r'D:\jack_doc\python_src\flyai\chinese_L-12_H-768_A-12'
BERT_CONFIG = os.path.join(BERT_PATH,"bert_config.json")
BERT_CKPT = os.path.join(BERT_PATH,'bert_model.ckpt')
VOCAB_FILE=os.path.join(BERT_PATH,"vocab.txt")


KERAS_MODEL_NAME = "my_BiLSTM_CRF.h5"
