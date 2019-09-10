#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : data_helper.py
# @Author: Zhan
# @Date  : 7/17/2019
# @Desc  : 预处理数据，创建字典、文本向量

import codecs
import json
import os

import numpy as np

from bert import tokenization
from path import *



# FILENAME_ORG = '带标签短信.txt'
# DICTFILE = 'vocab.dict'
# FILENAME = 'dev.csv'

tokenizer = tokenization.FullTokenizer(VOCAB_FILE)

def data_process(text_str):
    tokenizer = tokenization.FullTokenizer(VOCAB_FILE)
    tokens = tokenizer.tokenize(text_str)
    return tokens

def load_dict():
    word_dict_re = dict()
    word_dict = dict()
    with open(VOCAB_FILE, encoding='utf-8') as fin:
        for i, line in enumerate(fin.readlines()):
            word_dict[line[0:len(line)-1]] = i
    for k, v in word_dict.items():
        word_dict_re[v] = k
    return word_dict, word_dict_re

def sentence2ids(sentence,dict,max_seq_len=256):
    '''
    :param sentence:
    :param dict:
    :return:
    '''
    ids = []
    for word in data_process(sentence):
        if word in dict.keys():
            ids.append(dict[word])
        else:
            # 根据字典不同，选不同的标识符
            # ids.append(dict['_UNK_'])
            ids.append(dict['[UNK]'])

    if len(ids) < max_seq_len:
        # 根据字典不同，选不同的标识符
        # ids = ids + [dict['_PAD_'] for _ in range(max_seq_len - len(ids))]
        ids = ids + [dict['[PAD]'] for _ in range(max_seq_len - len(ids))]
    else:
        ids = ids[:max_seq_len]
    return ids

# def data_process_bert(text_str):
#     tokenizer = tokenization.FullTokenizer(VOCAB_FILE)
#     tokens = tokenizer.tokenize(text_str)
#     tokens.insert(0,'[CLS]')
#     tokens.append("[SEP]")
#     ids = tokenizer.convert_tokens_to_ids(tokens)
#     # ids = np.asarray(ids,dtype=np.int32)
#     return ids

def sentence2ids_bert(sentence,):
    '''
    :param sentence:
    :param dict:
    :return:
    '''
    tokens = tokenizer.tokenize(sentence)
    tokens.insert(0,'[CLS]')
    tokens.append("[SEP]")
    ids = tokenizer.convert_tokens_to_ids(tokens)
    return ids

#
# def generateDictFile(minNum=0):
#     '''
#     处理文本，生成字典文件
#     minNum： 词出现的最小次数 <minNum的词将直接丢弃
#     :return:
#     '''
#     # 初始化字典
#     dict = {"_PAD_": minNum+1, "_EOS_": minNum+1, "_SOS_": minNum+1, "_UNK_": minNum+1, }
#
#     fName = os.path.join(DATA_PATH,FILENAME)
#     with codecs.open(fName, encoding='utf-8', mode='r') as fi:
#         for i,line in enumerate(fi.readlines()):
#             if i == 0:
#                 continue
#             line = line.strip()
#             if len(line) > 2:
#                 for word in data_process(line[2:]):
#                     dict[word] = dict.setdefault(word,0) + 1
#
#     i = 0
#     rstDict = {}
#     for k,v in dict.items():
#         if v>=minNum:
#             rstDict[k] = i
#             i += 1
#
#     json_data = json.dumps(rstDict)
#
#     dictFile = os.path.join(DATA_PATH,DICTFILE)
#     with codecs.open(dictFile, encoding='utf-8', mode='w') as fo:
#         fo.write(json_data)
#
# def convert2csv():
#     '''
#     将原始短信文件转换为csv文件
#     :return:
#     '''
#     msgFile = os.path.join(DATA_PATH, FILENAME_ORG)
#     msgFile_t = os.path.join(DATA_PATH,FILENAME)
#
#     # windows 下 用 'w'模式。当文件存在，直接覆盖源文件。
#     with codecs.open(msgFile,encoding='utf-8',mode='r') as fi:
#         with codecs.open(msgFile_t,encoding='utf-8',mode='w') as fo:
#             fo.write('label,text'+os.linesep)
#             for line in fi.readlines():
#                 line = line.strip()
#                 if len(line) > 0:
#                     text = line[0] + ','
#                     text += line[1:].strip().replace(',', '，') + os.linesep
#                     fo.write(text)

if __name__ == '__main__':
    # #  将原始数据文件转换文csv文件
    # convert2csv()

    # generateDictFile(minNum=5)

    exit(0)






