#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : data_helper.py
# @Author: Zhan
# @Date  : 7/17/2019
# @Desc  : 预处理数据，创建字典、文本向量

import codecs

from keras_bert import Tokenizer
from path import *

token_dict = {}
with codecs.open(VOCAB_FILE, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)
tokenizer = Tokenizer(token_dict)

def load_dict():
    word_dict_re = dict()
    word_dict = dict()
    with open(VOCAB_FILE, encoding='utf-8') as fin:
        for i, line in enumerate(fin.readlines()):
            word_dict[line[0:len(line)-1]] = i
    for k, v in word_dict.items():
        word_dict_re[v] = k
    return word_dict, word_dict_re

def sentence2ids_bert(sentence,):
    '''
    :param sentence:
    :param dict:
    :return:
    '''
    tokens = tokenizer.tokenize(sentence)
    ids = tokenizer._convert_tokens_to_ids(tokens)
    return ids

# if __name__ == '__main__':
#     str = "中国 人民 戒饭军  休息休息 阿阿 阿  阿 ，  算求"
#
#     print(tokenizer.tokenize(str))



