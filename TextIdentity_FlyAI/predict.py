#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : predict.py
# @Author: Zhan
# @Date  : 12/3/2019
# @Desc  :

import pandas as pd
from flyai.dataset import Dataset
from model import Model

data = Dataset()
model = Model(data)

pdData = pd.read_csv(r'D:\jack_doc\python_src\flyai\TextIdentity_FlyAI\data\input\dev.csv')
for i, data in pdData.iterrows():
    p = model.predict(**data)
    print(p[0])