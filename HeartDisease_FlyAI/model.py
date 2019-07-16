# -*- coding: utf-8 -*
import numpy
import os
import torch
from flyai.model.base import Base
import xgboost as xgb

from path import MODEL_PATH

__import__('net', fromlist=["Net"])

Torch_MODEL_NAME = "model.pkl"


class Model(Base):
    # def __init__(self, xgbClassifier):
    #     self.xgbClassifier = xgbClassifier
    #
    # def predict(self, **data):
    #     prediction = self.xgbClassifier(data['dtest'])
    #     return prediction
    #
    # def predict_all(self, datas):
    #     return self.predict(datas)
    #
    # def save_model(self, model, path, name=Torch_MODEL_NAME, overwrite=False):
    #     super().save_model(model, path, name, overwrite)
    #     model.save_model(os.path.join(path, name))
    #

    def __init__(self, data):
        self.data = data
        if os.path.isfile(os.path.join(MODEL_PATH, Torch_MODEL_NAME)):
            self.xgbcls = xgb.Booster()
            self.xgbcls.load_model(os.path.join(MODEL_PATH, Torch_MODEL_NAME))
        else:
            self.xgbcls = None

    def predict(self, **data):
        if not self.xgbcls:
            self.xgbcls = xgb.Booster()
            self.xgbcls.load_model(os.path.join(MODEL_PATH, Torch_MODEL_NAME))

        x_data = self.data.predict_data(**data)
        x_data = xgb.DMatrix(x_data)
        outputs = self.xgbcls.predict(x_data)
        if outputs >= 0.5:
            prediction = 1 + 1
        else:
            prediction = 1

        # prediction = self.data.to_categorys(outputs)
        return prediction

    def predict_all(self, datas):
        labels = []
        # print(len(datas))
        i = 0
        for data in datas:
            prediction = self.predict(**data)
            labels.append(prediction)
            i += 1
        print("the length of test datas: %d" % i)
        return labels

    def batch_iter(self, x, y, batch_size=128):
        """生成批次数据"""
        data_len = len(x)
        num_batch = int((data_len - 1) / batch_size) + 1

        indices = numpy.random.permutation(numpy.arange(data_len))
        x_shuffle = x[indices]
        y_shuffle = y[indices]

        for i in range(num_batch):
            start_id = i * batch_size
            end_id = min((i + 1) * batch_size, data_len)
            yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]

    def save_model(self, xgbcls, path, name=Torch_MODEL_NAME, overwrite=False):
        super().save_model(xgbcls, path, name, overwrite)
        xgbcls.save_model(os.path.join(path, name))

