# -*- coding: utf-8 -*
import argparse
import torch
import torch.nn as nn
from flyai.dataset import Dataset
from torch.optim import Adam
import numpy as np
import xgboost as xgb


from model import Model
from net import Net
from path import MODEL_PATH

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=1000, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=32, type=int, help="batch size")
args = parser.parse_args()

def eval(preds_prob,y_test):
    preds = np.zeros(preds_prob.shape)
    preds[preds_prob >= 0.5] = 1
    train_accuracy = (preds == y_test).sum() / preds_prob.shape[0]

    return train_accuracy

# 训练并评估模型
data = Dataset()
x_train, y_train, x_test, y_test = data.next_batch(68,32)  # 读取数据
# read in data
dtrain = xgb.DMatrix(x_train,label = y_train)
dtest = xgb.DMatrix(x_test,label = y_test)

best_accuracy = 0
# specify parameters via map
param = {'max_depth': 5, 'eta':0.3, 'objective':'binary:logistic','verbosity':3 }
num_round = 10
# specify validations set to watch performance
watchlist = [(dtest, 'eval'), (dtrain, 'train')]
bst = xgb.train(param, dtrain, num_round,watchlist,verbose_eval=True)
# make prediction
preds_prob = bst.predict(dtest)

train_accuracy = eval(preds_prob, dtest.get_label())

if train_accuracy > best_accuracy:
    best_accuracy = train_accuracy
    model = Model(bst)
    model.save_model(bst, MODEL_PATH, overwrite=True)
    print("step %d, best accuracy %g" % (0, best_accuracy))