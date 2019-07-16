# -*- coding: utf-8 -*
import pickle
import os

import numpy as np
from flyai.model.base import Base
import xgboost as xgb

from path import MODEL_PATH

# __import__('net', fromlist=["Net"])

XGB_MODEL_NAME = "XGB_model.pkl"

def eval(preds_prob, y_test):
    preds = np.zeros(preds_prob.shape)
    preds[preds_prob >= 0.5] = 1
    train_accuracy = (preds == y_test).sum() / preds_prob.shape[0]

    return train_accuracy

class Model(Base):
    def __init__(self, data):
        self.data = data
        modelFile = os.path.join(MODEL_PATH, XGB_MODEL_NAME)
        if os.path.isfile(modelFile):
            self.clf = pickle.load(open(modelFile, "rb"))
        else:
            self.clf = None

    def predict(self, **data):
        if not self.clf:
            self.clf = pickle.load(
                open(
                    os.path.join(
                        MODEL_PATH,
                        XGB_MODEL_NAME),
                    "rb"))

        x_data = self.data.predict_data(**data)
        # x_data = xgb.DMatrix(x_data)
        outputs = self.clf.predict(x_data)
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

    def save_model(self, clf, path, name=XGB_MODEL_NAME, overwrite=False):
        super().save_model(clf, path, name, overwrite)
        pickle.dump(clf, open(os.path.join(path, name), "wb"))
        # xgbcls.save_model(os.path.join(path, name))
