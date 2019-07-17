# -*- coding: utf-8 -*
import pickle
import argparse

from flyai.dataset import Dataset
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

from model import Model, eval
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





# 训练并评估模型
data = Dataset(epochs=args.EPOCHS, batch=args.BATCH,)
model = Model(data)

x, y, x_test, y_test = data.get_all_processor_data()
#
# validateNum = 30
# x_train = x[0:x.shape[0]-validateNum,:]
# y_train = y[0:y.shape[0]-validateNum]
# x_test = x[-validateNum:,:]
# y_test = y[-validateNum:]

# x_train = x
# y_train = y

x_train = np.concatenate((x, x_test))
y_train = np.concatenate((y, y_test))

# region 本地测试用
# x_train = x
# y_train = y
# endregion

print("the length of train data: %d" % data.get_train_length())
print("the length of x_train: %d" % x_train.shape[0])
print("the length of x_test: %d" % x_test.shape[0])
# the length of train data: 162
# the length of x_train: 162
# the length of x_test: 54
# the length of test datas: 54
# x_train, y_train = data.get_all_validation_data()
# print(args.BATCH)
# print(args.EPOCHS)
# read in data
dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test, label=y_test)

# best_accuracy = 0
# specify parameters via map
# gamma/
etas = [0.001, 0.01, 0.1, 0.3, 0.5, 1]
max_depths = [2, 3, 5, 7, 10]
reg_lambdas = [0.001, 0.01, 0.1, 0.3, 0.5, 1, 7, 10, ]
reg_alphas = [0.001, 0.01, 0.1, 0.3, 0.5, 1, 7, 10, ]
num_rounds = [2, 5, 7, 10]
min_split_losss = [0.001, 0.01, 0.1, 1, 7, 10, 30, ]  # gamma

param_grid = {'learning_rate': [0.001, 0.01, 0.1, 0.3, 0.5, 1],
              'max_depth': [2, 3, 5, 7, 10],
              'reg_lambda': [0.001, 0.01, 0.1, 0.3, 0.5, 1, 7, 10, ],
              'reg_alpha': [0.001, 0.01, 0.1, 0.3, 0.5, 1, 7, 10, ],
              'num_round': [2, 5, 7, 10, 20],
              'gamma': [0, 0.0001, 0.001, 0.01, 0.1, 1, 7, 10, 30, ]
              }


# region 本地测试参数 ；
# param_grid = {'learning_rate': [0.1, 0.3],
#               'max_depth': [5,],
#               'reg_lambda': [ 0.3, 0.5,],
#               'reg_alpha': [0.3, ],
#               'num_round': [2,],
#               'gamma': [0.1, ]
#               }
# endregion

# etas = [0.001,0.5,]
# # max_depths = [2]
# # lambdas = [0.001,0.5, ]
# # alphas = [0.001]
# # num_rounds = [2, 7]

# watchlist = [(dtrain, 'train'), (dtest, 'eval')]

myscore = make_scorer(eval, greater_is_better=True)

bst = xgb.XGBClassifier(n_estimators=100, verbosity=0,
                        objective='binary:logistic', n_jobs= 5, subsample=0.7,)

clf = GridSearchCV(
    bst,
    param_grid,
    scoring=myscore,
    verbose=0,
    n_jobs=5,
    cv=6,
)

clf.fit(x_train, y_train)
print(clf.best_score_)
print(clf.best_params_)

model.save_model(clf, MODEL_PATH, overwrite=True)
