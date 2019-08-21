#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : model_tf_keras.py
# @Author: Zhan
# @Date  : 8/21/2019
# @Desc  :

import os

import tensorflow as tf
from tensorflow.keras.models import save_model, load_model
from tensorflow.keras.applications.densenet import preprocess_input
from flyai.model.base import Base
from path import MODEL_PATH

KERAS_MODEL_NAME = "my_densenet.h5"

class Model(Base):
    def __init__(self, dataset):
        self.dataset = dataset
        self.model_path = os.path.join(MODEL_PATH, KERAS_MODEL_NAME)
        self.model = None
        if os.path.isfile(self.model_path):
            self.model = load_model(self.model_path)

    '''
    评估一条数据
    '''

    def predict(self, **data):
        if self.model is None:
            self.model = load_model(self.model_path)
        # 获取需要预测的图像数据， predict_data 方法默认会去调用 processor.py 中的 input_x 方法
        x_data = self.dataset.predict_data(**data)
        x_data = preprocess_input(x_data,)

        predict = self.model.predict(x_data)
        # 将预测数据转换成对应标签  to_categorys 会去调用 processor.py 中的 output_y 方法
        prediction = self.dataset.to_categorys(predict)
        return prediction

    '''
    评估的时候会调用该方法实现评估得分
    '''

    def predict_all(self, datas):
        if self.model is None:
            self.model = load_model(self.model_path)
        predictions = []
        for data in datas:
            prediction = self.predict(**data)
            predictions.append(prediction)
        return predictions

    '''
    保存模型的方法
    '''
    def save_model(self, model, overwrite=False):
        super().save_model(model, MODEL_PATH, KERAS_MODEL_NAME, overwrite)
        ### 加入模型保存代码
        if not os.path.exists(MODEL_PATH):
            os.makedirs(MODEL_PATH)
        save_model(model,self.model_path, overwrite, include_optimizer=False)

