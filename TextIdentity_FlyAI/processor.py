# -*- coding: utf-8 -*

import os
import numpy
import json
import jieba
from flyai.processor.base import Base
from path import DATA_PATH


def load_dict(dict_file):
    if not os.path.exists(dict_file):
        print("[ERROR] load_dict failed! | The params: {}".format(dict_file))
        return None
    with open(dict_file, 'r', encoding='utf8') as f:
        tmp_dict = json.load(f)

    count = 0
    data2id, id2data = dict(), dict()
    for key1, value1 in tmp_dict.items():
        data2id[key1] = count
        count += 1

    for key2, value2 in data2id.items():
        id2data[value2] = key2

    return data2id, id2data


def data_process(text_line, data_dict, max_seq_len=124):
    text_line = jieba.lcut(text_line)
    que_len = len(text_line)
    que_list = list()
    for i in range(len(text_line)):
        if text_line[i] in data_dict.keys():
            que_list.append(data_dict[text_line[i]])
        else:
            que_list.append(data_dict['<UNK>'])

    if len(que_list) < max_seq_len:
        que_list += [data_dict['<PAD>'] for _ in range(max_seq_len - len(que_list))]
    else:
        que_list = que_list[:max_seq_len]
    return que_list, que_len


'''
把样例项目中的processor.py件复制过来替换即可
'''


class Processor(Base):
    '''
    参数为csv中作为输入x的一条数据，该方法会被dataset.next_train_batch()
    和dataset.next_validation_batch()多次调用。可在该方法中做数据增强
    该方法字段与app.yaml中的input:->columns:对应
    '''
    def __init__(self):
        super(Processor, self).__init__()
        # 加载词频字典数据
        self.que2id, self.id2que = load_dict(os.path.join(DATA_PATH, 'que_fr.dict'))
        self.ans2id, self.id2ans = load_dict(os.path.join(DATA_PATH, 'ans_fr.dict'))

    '''
    参数为csv中作为输入x的一条数据，该方法会被dataset.next_train_batch()
    和dataset.next_validation_batch()多次调用。评估的时候会调用该方法做数据处理
    该方法字段与app.yaml中的input:->columns:对应
    '''
    def input_x(self, usr_text, ans_comment):
        usr_list = data_process(usr_text, self.que2id)
        ans_list = data_process(ans_comment, self.ans2id)
        return usr_list, ans_list

    '''
    参数为csv中作为输入y的一条数据，该方法会被dataset.next_train_batch()
    和dataset.next_validation_batch()多次调用。
    该方法字段与app.yaml中的output:->columns:对应
    '''
    def input_y(self, label):
        # 为避免出现非数值形式的label，可以强制转化成int
        return int(label)

    '''
    输出的结果，会被dataset.to_categorys(data)调用
    '''
    def output_y(self, data):
        # 返回分类结果0/1
        return numpy.argmax(data,axis=-1)