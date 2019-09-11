#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : path.py
# @Author: Zhan
# @Date  : 7/18/2019
# @Desc  : 数据、模型、字典等文件路径

import sys
import os

from bert import modeling

from flyai.utils import remote_helper

cPath = os.getcwd()
# 训练数据的路径
DATA_PATH = os.path.join(cPath, 'data', 'input')
# 模型保存的路径
MODEL_PATH = os.path.join(cPath, 'data', 'output', 'model')
# 训练log的输出路径
LOG_PATH = os.path.join(cPath, 'data', 'output', 'logs')

# 必须使用该方法下载模型，然后加载
BERT_PATH = remote_helper.get_remote_date("https://www.flyai.com/m/chinese_L-12_H-768_A-12.zip")
print('BERT_PATH:{}'.format(BERT_PATH))
# BERT_PATH = r'D:\jack_doc\python_src\flyai\chinese_L-12_H-768_A-12'
BERT_CONFIG = modeling.BertConfig.from_json_file(os.path.join(BERT_PATH,"bert_config.json"))
BERT_CKPT = os.path.join(BERT_PATH,'bert_model.ckpt')
VOCAB_FILE=os.path.join(BERT_PATH,"vocab.txt")


TENSORFLOW_MODEL_DIR = "dpNet.ckpt"