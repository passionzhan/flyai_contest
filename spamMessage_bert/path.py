#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : path.py
# @Author: Zhan
# @Date  : 7/18/2019
# @Desc  : 数据、模型、字典等文件路径

import sys
import os

cPath = os.getcwd()
# 训练数据的路径
DATA_PATH = os.path.join(cPath, 'data', 'input')
# 模型保存的路径
MODEL_PATH = os.path.join(cPath, 'data', 'output', 'model')
# 训练log的输出路径
LOG_PATH = os.path.join(cPath, 'data', 'output', 'logs')