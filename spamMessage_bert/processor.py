#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : processor.py
# @Author: Zhan
# @Date  : 7/17/2019
# @Desc  : 数据处理类

from data_helper import *
from flyai.processor.base import Base

MAX_LEN = 128


class Processor(Base):
    def __init__(self):
        super(Processor, self).__init__()
        self.word_dict, self.word_dict_re = load_dict()

    def getWordsCount(self):
        return len(self.word_dict)

    def input_x(self, text):
        '''
        参数为csv中作为输入x的一条数据，该方法会被dataset.next_train_batch()
        和dataset.next_validation_batch()多次调用。可在该方法中做数据增强
        该方法字段与app.yaml中的input:->columns:对应
        '''
        # sent_ids = sentence2ids(text, self.word_dict)
        sent_ids = sentence2ids_bert(text,)
        return sent_ids

    def input_y(self, label):
        '''
        参 数为csv中作为输入y的一条数据，该方法会被dataset.next_train_batch()
        和dataset.next_validation_batch()多次调用。
        该方法字段与app.yaml中的output:->columns:对应
        '''
        # 0 - 非垃圾短信
        # 1 - 垃圾短信
        return int(label)

    def output_y(self, data):
        '''
        输出的结果，会被dataset.to_categorys(data)调用
        '''

        return data[0]










