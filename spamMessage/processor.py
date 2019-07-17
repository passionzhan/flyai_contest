#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : processor.py
# @Author: Zhan
# @Date  : 7/17/2019
# @Desc  : 预处理数据，创建字典、文本向量

import codecs
import json

import jieba
FILENAME = '带标签短信.txt'
DICTFILE = 'dict.txt'

def load_dict():
    word_dict_re = dict()
    with open(DICTFILE, encoding='utf-8') as fin:
        word_dict = json.load(fin)
    for k, v in word_dict.items():
        word_dict_re[v] = k
    return word_dict, word_dict_re

def sentence2ids(sentence,dict):
    '''
    :param sentence:
    :param dict:
    :return:
    '''
    ids = []
    for word in jieba.lcut(sentence):
        ids.append(dict[word])
    return ids

# def generate

def generateDictFile(minNum=0):
    '''
    处理文本，生成字典文件
    minNum： 词出现的最小次数 <minNum的词将直接丢弃
    :return:
    '''
    # 初始化字典
    dict = {"_PAD_": minNum+1, "_EOS_": minNum+1, "_SOS_": minNum+1, "_UNK_": minNum+1, }

    with codecs.open(FILENAME,encoding='utf-8',mode='r') as fi:
        for line in fi.readlines():
            line = line.strip()
            if len(line) > 0:
                text = line[1:].strip().replace('\s+', ' ', 3)
                #  默认精确模式分词
                for word in jieba.lcut(text,):
                    dict[word] = dict.setdefault(word,0) + 1

    i = 0
    rstDict = {}
    for k,v in dict.items():
        if v>=minNum:
            rstDict[k] = i
            i += 1

    json_data = json.dumps(rstDict)

    with codecs.open(DICTFILE, encoding='utf-8', mode='w') as fo:
        fo.write(json_data)


if __name__ == '__main__':
    generateDictFile(minNum=5)









