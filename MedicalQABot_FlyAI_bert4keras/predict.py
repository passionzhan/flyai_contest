# -*- coding: utf-8 -*
'''
实现模型的调用
'''
import pandas as pd
from flyai.dataset import Dataset
from model import Model

data = Dataset()
model = Model(data)

pdData = pd.read_csv(r'D:\jack_doc\python_src\flyai\data\MedicalQABot\test_tmp.csv')
# model.predict_all(pdData.source)
# model.seq2seqModel.load_weights(model.model_path)
# processor=Processor()
for text in pdData.que_text:
    p = model.predict(que_text=text)

    # p = model.predict(load_weights = True,que_text="孕妇检查四维彩超的时候医生会给家属进去看吗")
    print(p)
