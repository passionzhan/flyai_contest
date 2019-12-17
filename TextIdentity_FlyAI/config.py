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

max_seq_len                 = 512

learning_rate               = 1e-3
MAX_GRAD_NORM               = 1.0
VAL_STEPS_PER_VAL           = 30
VAL_FREQUENCY               = 100
