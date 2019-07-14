# -*- coding: utf-8 -*
import numpy
import os
import torch
from flyai.model.base import Base

from path import MODEL_PATH

__import__('net', fromlist=["Net"])

Torch_MODEL_NAME = "model.pkl"


class Model(Base):
    def __init__(self, xgbClassifier):
        self.xgbClassifier = xgbClassifier

    def predict(self, **data):
        prediction = self.xgbClassifier(data['dtest'])
        return prediction

    def predict_all(self, datas):
        return self.predict(datas)

    def save_model(self, model, path, name=Torch_MODEL_NAME, overwrite=False):
        super().save_model(model, path, name, overwrite)
        model.save_model(os.path.join(path, name))
