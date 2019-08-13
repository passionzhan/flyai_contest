# -*- coding: utf-8 -*
import sys

import os

cPath = os.getcwd()
# 训练数据的路径
DATA_PATH = os.path.join(cPath, 'data', 'input')
# 模型保存的路径
MODEL_PATH = os.path.join(cPath, 'data', 'output', 'model')
# 训练log的输出路径
LOG_PATH = os.path.join(cPath, 'data', 'output', 'logs')
