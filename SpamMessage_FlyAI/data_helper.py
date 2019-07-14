# -*- coding: utf-8 -*-

import os
import json
import jieba
import pandas as pd
from path import DATA_PATH
from collections import Counter
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def data_process(text_str):
    if len(text_str) == 0:
        print('[ERROR] data_process failed! | The params: {}'.format(text_str))
        return None
    text_str = text_str.strip().replace('\s+', ' ', 3)
    return jieba.lcut(text_str)


def load_dict():
    char_dict_re = dict()
    dict_path = os.path.join(DATA_PATH, 'words.dict')
    with open(dict_path, encoding='utf-8') as fin:
        char_dict = json.load(fin)
    for k, v in char_dict.items():
        char_dict_re[v] = k
    return char_dict, char_dict_re


def word2id(text_str, word_dict, max_seq_len=128):
    if len(text_str) == 0 or len(word_dict) == 0:
        print('[ERROR] word2id failed! | The params: {} and {}'.format(text_str, word_dict))
        return None

    sent_list = data_process(text_str)
    sent_ids = list()
    for item in sent_list:
        if item in word_dict:
            sent_ids.append(word_dict[item])
        else:
            sent_ids.append(word_dict['_UNK_'])

    if len(sent_ids) < max_seq_len:
        sent_ids = sent_ids + [word_dict['_PAD_'] for _ in range(max_seq_len - len(sent_ids))]
    else:
        sent_ids = sent_ids[:max_seq_len]
    return sent_ids


if __name__ == "__main__":

    exit(0)