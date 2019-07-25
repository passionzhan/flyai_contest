#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : dataset.py
# @Author: Zhan
# @Date  : 7/24/2019
# @Desc  :

import os
import gevent
from gevent import pool, monkey

from pandas.io.parsers import read_csv
from flyai.utils.yaml_helper import Yaml
import numpy

from path import DATA_PATH
from processor import Processor

monkey.patch_all(ssl=False, thread=False)

pool = pool.Pool(128)


class Dataset(object):
    def __init__(self, train_batch=32, val_batch=64, split_ratio = 0.9, transformation=None, ):
        self.data = read_csv(os.path.join(DATA_PATH, "message.csv"))
        self.data_len = self.data.shape[0]
        self.__train_page = 0
        self.__val_page = 0
        self.train_num = (int)(self.data_len * split_ratio)
        self.val_num = self.data_len - self.train_num
        self.__train_BATCH = train_batch
        self.__val_BATCH = val_batch

        self.__model = Yaml().processor()
        clz = self.__model['processor']
        self.processor = self.create_instance("processor", clz)

    def next_train_batch(self):
        maxIdx = min(self.data_len,(self.__val_page+1)*self.__train_BATCH)
        train = self.data[self.__train_page * self.__train_BATCH: maxIdx]
        if maxIdx == self.train_num:
            self.__train_page = 0
        else:
            self.__train_page += 1

        x_train, y_train = self.__get_data(train)
        x_train = self.__processor_x(x_train)
        y_train = self.__processor_y(y_train)

        return x_train, y_train

    def next_val_batch(self):
        maxIdx = min(self.data_len,(self.__val_page+1)*self.__val_BATCH + self.train_num)
        val = self.data[self.__val_page * self.__val_BATCH + self.train_num: maxIdx]
        if maxIdx == self.data_len:
            self.__val_page = 0
        else:
            self.__train_page += 1

        x_val, y_val = self.__get_data(val)
        x_val = self.__processor_x(x_val)
        y_val = self.__processor_y(y_val)

        return x_val, y_val

    def get_all_train(self):
        train = self.data[0:self.train_num]
        x_train, y_train = self.__get_data(train)
        x_train = self.__processor_x(x_train)
        y_train = self.__processor_y(y_train)

        return x_train, y_train

    def get_all_val(self):
        val = self.data[self.train_num:]
        x_val, y_val = self.__get_data(val)
        x_val = self.__processor_x(x_val)
        y_val = self.__processor_y(y_val)

        return x_val, y_val

    def to_categorys(self, predict):
        return self.get_method_list(self.processor, self.__model['output_y'], predict)

    def __get_data(self, data):
        yaml = Yaml()
        x_names = yaml.get_input_names()
        y_names = yaml.get_output_names()
        x_data = data[x_names]
        y_data = data[y_names]
        x_data = x_data.to_dict(orient='records')
        y_data = y_data.to_dict(orient='records')
        return x_data, y_data

    def __processor_x(self, x_datas):
        threads = []
        for data in x_datas:
            threads.append(pool.spawn(self.get_method_dict, self.processor, self.__model['input_x'], **data))
        gevent.joinall(threads)
        processors = []
        init = False
        processor_len = 0
        for i, g in enumerate(threads):
            processor = g.value
            if not isinstance(processor, tuple):
                processor_len = 1
            if processor_len == 1:
                processors.append(numpy.array(processor))
            else:
                if not init:
                    processors = [[] for i in range(len(processor))]
                    init = True
                index = 0
                for item in processor:
                    processors[index].append(numpy.array(item))
                    index += 1
        if processor_len == 1:
            return numpy.concatenate([processors], axis=0)
        else:
            list = []
            for column in processors:
                list.append(numpy.concatenate([column], axis=0))
            return list

    def __processor_y(self, y_datas):
        threads = []
        for data in y_datas:
            threads.append(pool.spawn(self.get_method_dict, self.processor, self.__model['input_y'], **data))
        gevent.joinall(threads)

        processors = []
        init = False
        processor_len = 0

        for i, g in enumerate(threads):
            processor = g.value
            if not isinstance(processor, tuple):
                processor_len = 1
            if processor_len == 1:
                processors.append(numpy.array(processor))
            else:
                if not init:
                    processors = [[] for i in range(len(processor))]
                    init = True
                index = 0
                for item in processor:
                    processors[index].append(numpy.array(item))
                    index += 1
        if processor_len == 1:
            return numpy.concatenate([processors], axis=0)
        else:
            list = []
            for column in processors:
                list.append(numpy.concatenate([column], axis=0))
            return list

    def get_method_dict(self, clz, method_name, **args):
        m = getattr(clz, method_name)
        return m(**args)

    def get_method_list(self, clz, method_name, *args):
        m = getattr(clz, method_name)
        return m(*args)

    def create_instance(self, module_name, class_name, *args, **kwargs):
        module_meta = __import__(module_name, globals(), locals(), [class_name])
        class_meta = getattr(module_meta, class_name)
        return class_meta(*args, **kwargs)

if __name__ == '__main__':
    dataset = Dataset(64,128)
    x_train, y_train = dataset.next_train_batch()
    x_val, y_val = dataset.next_val_batch()

    print(dataset.data)


