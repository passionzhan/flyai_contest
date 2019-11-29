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


QUE_DICT_FILE = os.path.join(DATA_PATH, 'que.dict')
ANS_DICT_FILE = os.path.join(DATA_PATH, 'ans.dict')

# 必须使用该方法下载模型，然后加载
# BERT_PATH       = remote_helper.get_remote_date("https://www.flyai.com/m/chinese_L-12_H-768_A-12.zip")
BERT_PATH         = remote_helper.get_remote_date('https://www.flyai.com/m/chinese_wwm_ext_pytorch.zip')
print('after get_remote_date BERT_PATH:{}'.format(BERT_PATH))
BERT_PATH_TMP     = os.path.dirname(BERT_PATH)

BERT_PATH = os.path.join(BERT_PATH_TMP,"chinese_wwm_ext_pytorch_bert")
print('BERT_PATH:{}'.format(BERT_PATH))

if not os.path.exists(BERT_PATH):
    os.makedirs(BERT_PATH)

#获取目录列表
list_dir = os.listdir(BERT_PATH_TMP)
#打印目录列表
for temp in list_dir:
    temp_long = os.path.join(BERT_PATH_TMP,temp)
    if not temp.endswith(".zip") and os.path.isfile(temp_long):
        print(temp_long)
        copyfile(temp_long, os.path.join(BERT_PATH,temp))

# 重命名
os.rename(os.path.join(BERT_PATH,"bert_config.json"),os.path.join(BERT_PATH,"config.json"))


# BERT_PATH = r'D:\jack_doc\python_src\flyai\data\bert_chinese_wwm_ext_pytorch'
# BERT_CONFIG     = os.path.join(BERT_PATH,"bert_config.json")
# BERT_CKPT       = os.path.join(BERT_PATH,'bert_model.ckpt')
# VOCAB_FILE      = os.path.join(BERT_PATH,"vocab.txt")
#
# MY_VOCAB_FILE   = os.path.join(DATA_PATH, "MY_VOCAB.json")

QA_MODEL_DIR    = "medicalQABot_bert"

