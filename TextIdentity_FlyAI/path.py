# -*- coding: utf-8 -*
import sys
import os
from shutil import copyfile

from flyai.utils import remote_helper

# 训练数据的路径
DATA_PATH = os.path.join(sys.path[0], 'data', 'input')
# 模型保存的路径
MODEL_PATH = os.path.join(sys.path[0], 'data', 'output', 'model')
# 训练log的输出路径
LOG_PATH = os.path.join(sys.path[0], 'data', 'output', 'logs')


# 必须使用该方法下载模型，然后加载
ALBERT_PATH = remote_helper.get_remote_date('https://www.flyai.com/m/pytorch_albert_small_zh_google.zip')
ALBERT_PATH = os.path.join(os.path.dirname(ALBERT_PATH), "albert_small_zh_google_pytorch")

# ALBERT_PATH         = r"D:\jack_doc\python_src\flyai\data\albert_small_zh_google_pytorch"
TEXTIDENTITY_MODEL_DIR    = "textidentity_albert"
