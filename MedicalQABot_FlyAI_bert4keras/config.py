#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   config.py    
@Contact :   9824373@qq.com
@License :   (C)Copyright 2017-2018, Zhan
@Desc    :     
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/9/22 17:14   zhan      1.0        超参数配置文件
'''
# import os
import json

from path import *

def load_dict():
    with open(QUE_DICT_FILE, 'r', encoding='UTF-8') as gf:
        que_dict = json.load(gf)
    with open(ANS_DICT_FILE, 'r', encoding='UTF-8') as pf:
        ans_dict = json.load(pf)

    # 将 '_pad_' 单词作为 第 0 个单词
    que_idx2word = {v: k for k, v in que_dict.items()}
    ans_idx2word = {v: k for k, v in ans_dict.items()}
    pad_idx = que_dict['_pad_']
    word0 = que_idx2word[0]
    que_dict['_pad_'], que_dict[word0] = 0, pad_idx
    que_idx2word[0], que_idx2word[pad_idx] = '_pad_', word0
    pad_idx = ans_dict['_pad_']
    word0 = ans_idx2word[0]
    ans_dict['_pad_'], ans_dict[word0] = 0, pad_idx
    ans_idx2word[0], ans_idx2word[pad_idx] = '_pad_', word0
    return que_dict, ans_dict


# region 词典构建
que_dict, ans_dict = load_dict()
# ans_sos_idx = ans_dict['_sos_']

# # region 合并问题与答案词典
# for k, v in ans_dict.items():
#     que_dict.setdefault(k,len(que_dict))
# ans_dict = que_dict
# # endregion
# endregion

max_que_seq_len             = 128

max_ans_seq_len_predict     = 128
max_seq_len                 = 256
hide_dim                    = 512
eDim                        = 200
# Embedding Size
encoding_embedding_size     = 64
decoding_embedding_size     = 64
# Learning Rate
learning_rate               = 0.01
DROPOUT_RATE                = 0.2

IGNORE_WORD_IDX             = 102

encode_vocab_size           = len(que_dict)
decode_vocab_size           = len(ans_dict)