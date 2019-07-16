from flyai.dataset import Dataset

from model import Model

data = Dataset()
model = Model(data)
p = model.predict(age=53, sex=1, cp=3, trestbps=130, chol=246, fbs=1, restecg=2, thalach=173, exang=0, oldpeak=0.0,
                  slope=1, ca=3, thal=3)
print(p)
