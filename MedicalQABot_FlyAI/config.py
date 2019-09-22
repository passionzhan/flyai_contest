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

from data_helper import *

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
max_ans_seq_len             = 128
hide_dim                    = 512
eDim                        = 200
# Embedding Size
encoding_embedding_size     = 64
decoding_embedding_size     = 64
# Learning Rate
learning_rate               = 0.001
DROPOUT_RATE                = 0.2

encode_vocab_size           = len(que_dict)
decode_vocab_size           = len(ans_dict)