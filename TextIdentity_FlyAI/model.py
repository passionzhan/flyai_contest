# -*- coding: utf-8 -*
import numpy
import os
import torch
from flyai.model.base import Base

from path import *

__import__('net', fromlist=["Net"])

TORCH_MODEL_NAME = "model.pkl"

def getDevive():# Set up training device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device

def get_NumofGPU():#
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
    else:
        n_gpu = 0
    return n_gpu

class Model(Base):
    def __init__(self, data):
        self.data = data
        self.net_path = os.path.join(MODEL_PATH, TEXTIDENTITY_MODEL_DIR)
        if os.path.exists(self.net_path):
            print('加载训练好的模型：')
            self.tokenizer = AlbertTokenizer.from_pretrained(self.net_path)
            self.net = AlbertForSequenceClassification.from_pretrained(self.net_path)
            print('加载训练好的模型结束')
        else:
            self.tokenizer = AlbertTokenizer.from_pretrained(ALBERT_PATH)
            self.net = AlbertForSequenceClassification.from_pretrained(ALBERT_PATH)

    def predict(self, **data):
        x_data = self.data.predict_data(**data)
        x_data = torch.from_numpy(x_data)
        outputs = self.net(x_data)
        prediction = outputs.data.numpy()
        prediction = self.data.to_categorys(prediction)
        return prediction

    def predict_all(self, datas):
        labels = []
        for data in datas:
            labels.append(self.predict(**data)[0])
        return labels

    def save_model(self,):
        super().save_model(None, self.net_path, "albert_pytorch.bin", True)
        self.net.save_pretrained(self.net_path)
        self.tokenizer.save_pretrained(self.net_path)