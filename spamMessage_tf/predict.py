# -*- coding: utf-8 -*
from flyai.dataset import Dataset
from model import Model

from processor import Processor

# 数据获取辅助类
dataset = Dataset()
# 模型操作辅助类
model = Model(dataset)

result = model.predict(text="您好！我们这边是施华洛世奇鄞州万达店！您是我们尊贵的会员，特意邀请您参加我们x.x-x.x的三八女人节活动！满xxxx元享晶璨花漾丝巾")
print(result)

tData = dataset.get_all_data()
preds = model.predict_all(tData[0])

y_test = []
for label in tData[1]:
    y_test.append(label['label'])

rCount = 0.0
for i in range(0,len(preds)):
    if preds[i] == y_test[i]:
        rCount += 1.

test_acc = rCount / len(preds)

print('accuracy %g' % test_acc)