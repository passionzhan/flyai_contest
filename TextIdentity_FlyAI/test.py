#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : test.py
# @Author: Zhan
# @Date  : 12/2/2019
# @Desc  :

from flyai.dataset import Dataset

from utilities import data_split

dataset = Dataset(epochs=2, batch=10)

# x_train, y_train, x_val, y_val = dataset.get_all_data()

x_train, y_train, x_val, y_val = data_split(dataset,val_ratio=0.1)