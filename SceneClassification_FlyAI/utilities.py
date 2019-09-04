#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : utilities.py
# @Author: Zhan
# @Date  : 9/4/2019
# @Desc  :

import numpy as np
from flyai.dataset import Dataset

def label_smoothing(inputs, epsilon=0.1):
    '''
    标签平滑，
    :param inputs: 输入标签，最后一维是one-hot 表示
    :param epsilon:
    :return:
    '''
    K = inputs.shape[-1] # number of class
    return ((1 - epsilon) * inputs) + (epsilon / K)


def data_split(dataset,val_ratio):
    x_train, y_train, x_val, y_val = dataset.get_all_data()
    x_data = np.concatenate((x_train, x_val))
    y_data = np.concatenate((y_train, y_val))
    val_ratio = 0.1
    train_len = int(x_data.shape[0] * (1 - val_ratio))
    x_train = x_data[0:train_len]
    y_train = y_data[0:train_len]
    x_val = x_data[train_len:]
    y_val = y_data[train_len:]
    return x_train, y_train, x_val, y_val

