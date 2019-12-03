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

min_count                   = 5
max_seq_len                 = 512
max_que_seq_len             = 512
max_ans_seq_len             = 256
max_ans_seq_len_predict     = 128

# max_seq_len                 = 256
hide_dim                    = 512
eDim                        = 200
# Embedding Size
encoding_embedding_size     = 64
decoding_embedding_size     = 64
# Learning Rate
learning_rate               = 1e-4
DROPOUT_RATE                = 0.2

IGNORE_WORD_IDX             = 102

FIRST_VALIDED_TOKEN           = '[SEP]'
# encode_vocab_size           = len(que_dict)
# decode_vocab_size           = len(ans_dict)

sentence_end_token          = '[SEP]'
sentence_sep_token          = '[SEP]'


MAX_GRAD_NORM               = 1.0
VAL_STEPS_PER_VAL           = 2
VAL_FREQUENCY               = 3
