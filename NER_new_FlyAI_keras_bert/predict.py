# -*- coding: utf-8 -*
'''
实现模型的调用
'''
from flyai.dataset import Dataset
import pandas as pd

from model import Model
from path import MODEL_PATH
from  processor import Processor

data = Dataset()
model = Model(data)

pdData = pd.read_csv(r'D:\jack_doc\python_src\flyai\data\NER_new\test.csv')
# model.predict_all(pdData.source)
model.ner_model.load_weights(model.model_path)
processor=Processor()
for text in pdData.source:
    p = model.predict(processor, load_weights=False,source=text)
    # p = model.predict(processor, load_weights=True,
    #               source="新华社 北京 9 月 11 日电 第二十二届 国际 检察官 联合会 年会 暨 会员 代表大会 11 日 上午 在 北京 开幕 。 国家 主席 习近平 发来 贺信 ， 对 会议 召开 表示祝贺 。")
    print(p)
