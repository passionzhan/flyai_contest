# -*- coding: utf-8 -*
'''
实现模型的调用
'''
from flyai.dataset import Dataset
from model import QAModel

data = Dataset()
model = QAModel(data)

p = model.predict(load_weights = True,que_text="孕妇检查四维彩超的时候医生会给家属进去看吗")
print(p)
