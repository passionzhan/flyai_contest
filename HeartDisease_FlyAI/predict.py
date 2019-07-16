from flyai.dataset import Dataset

from model import Model

data = Dataset()
model = Model(data)
# p = model.predict(age=53, sex=1, cp=3, trestbps=130, chol=246, fbs=1, restecg=2, thalach=173, exang=0, oldpeak=0.0,
#                   slope=1, ca=3, thal=3)
# print(p)

tData = data.get_all_data()
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




