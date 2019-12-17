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


# # 必须使用该方法下载模型，然后加载
# BERT_PATH = remote_helper.get_remote_date('https://www.flyai.com/m/albert_tiny_zh_google_pytorch.zip')
# BERT_PATH = os.path.join(os.path.dirname(BERT_PATH), "albert_tiny_zh_google_pytorch")

BERT_PATH = remote_helper.get_remote_date('https://www.flyai.com/m/chinese_wwm_ext_pytorch.zip')
BERT_PATH = os.path.dirname(BERT_PATH)
copyfile(os.path.join(BERT_PATH, "bert_config.json"), os.path.join(BERT_PATH, "config.json"))


# BERT_PATH         = r"D:\jack_doc\python_src\flyai\data\bert_chinese_wwm_ext_pytorch"
TEXTIDENTITY_MODEL_DIR    = "textidentity_bert"
