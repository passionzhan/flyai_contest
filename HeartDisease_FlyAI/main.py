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
parser.add_argument(
    "-e",
    "--EPOCHS",
    default=1000,
    type=int,
    help="train epochs")
parser.add_argument("-b", "--BATCH", default=32, type=int, help="batch size")
args = parser.parse_args()


def eval(preds_prob, y_test):
    preds = np.zeros(preds_prob.shape)
    preds[preds_prob >= 0.5] = 1
    train_accuracy = (preds == y_test).sum() / preds_prob.shape[0]

    return train_accuracy


# 训练并评估模型
data = Dataset()
model = Model(data)

x_train, y_train, x_test, y_test = data.get_all_processor_data()
# read in data
dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test, label=y_test)

best_accuracy = 0
# specify parameters via map
etas = [0.001, 0.01, 0.1, 0.3, 0.5, 1]
max_depths = [2, 3, 5, 7, 10]
lambdas = [0.001, 0.01, 0.1, 0.3, 0.5, 1, 7, 10, ]
alphas = [0.001, 0.01, 0.1, 0.3, 0.5, 1, 7, 10, ]
num_rounds = [2, 5, 7, 10]

# etas = [0.001,0.5,]
# max_depths = [2]
# lambdas = [0.001,0.5, ]
# alphas = [0.001]
# num_rounds = [2, 7]

watchlist = [(dtest, 'eval'), (dtrain, 'train')]

for eta in etas:
    for max_depth in max_depths:
        for lambda_2 in lambdas:
            for alpha in alphas:
                for num_round in num_rounds:
                    param = {
                        'max_depth': max_depth,
                        'eta': eta,
                        'lambda': lambda_2,
                        'alpha': alpha,
                        'objective': 'binary:logistic',
                        'verbosity': 3}

                    # specify validations set to watch performance
                    bst = xgb.train(
                        param,
                        dtrain,
                        num_round,
                        watchlist,
                        verbose_eval=True)
                    preds_prob = bst.predict(dtest)
                    train_accuracy = eval(preds_prob, dtest.get_label())

                    if train_accuracy > best_accuracy:
                        best_accuracy = train_accuracy
                        model.save_model(bst, MODEL_PATH, overwrite=True)
                        print(
                            "parameters: eta: %g, max_depth: %g, lambda_2: %g, alpha: %g, num_round: %g." %
                            (eta, max_depth, lambda_2, alpha, num_round))
                        print("best accuracy %s" % best_accuracy)
