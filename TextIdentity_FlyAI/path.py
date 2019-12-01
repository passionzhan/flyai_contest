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
ALBERT_PATH         = remote_helper.get_remote_date('https://www.flyai.com/m/chinese_wwm_ext_pytorch.zip')
print('after get_remote_date ALBERT_PATH:{}'.format(ALBERT_PATH))
ALBERT_PATH_TMP     = os.path.dirname(ALBERT_PATH)

ALBERT_PATH = os.path.join(ALBERT_PATH_TMP, "chinese_wwm_ext_pytorch_bert")
print('BERT_PATH:{}'.format(ALBERT_PATH))

if not os.path.exists(ALBERT_PATH):
    os.makedirs(ALBERT_PATH)

#获取目录列表
list_dir = os.listdir(ALBERT_PATH_TMP)
#打印目录列表
for temp in list_dir:
    temp_long = os.path.join(ALBERT_PATH_TMP, temp)
    if not temp.endswith(".zip") and os.path.isfile(temp_long):
        print(temp_long)
        copyfile(temp_long, os.path.join(ALBERT_PATH, temp))

# 重命名
os.rename(os.path.join(ALBERT_PATH, "bert_config.json"), os.path.join(ALBERT_PATH, "config.json"))

TEXTIDENTITY_MODEL_DIR    = "textidentity_albert"
