# -*- coding: utf-8 -*
from flyai.dataset import Dataset
from model import Model

# 数据获取辅助类
dataset = Dataset()
# 模型操作辅助类
model = Model(dataset)

result = model.predict(text="您好！我们这边是施华洛世奇鄞州万达店！您是我们尊贵的会员，特意邀请您参加我们x.x-x.x的三八女人节活动！满xxxx元享晶璨花漾丝巾")
print(result)