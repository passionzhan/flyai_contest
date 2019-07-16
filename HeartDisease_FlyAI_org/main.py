# -*- coding: utf-8 -*
import argparse
import torch
import torch.nn as nn
from flyai.dataset import Dataset
from torch.optim import Adam

from model import Model
from net import Net
from path import MODEL_PATH

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=10, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=32, type=int, help="batch size")
args = parser.parse_args()


def eval(model, x_test, y_test):
    network.eval()
    batch_eval = model.batch_iter(x_test, y_test)
    total_acc = 0.0
    data_len = len(x_test)
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        outputs = network(x_batch)
        _, prediction = torch.max(outputs.data, 1)
        correct = (prediction == y_batch).sum().item()
        acc = correct / batch_len
        total_acc += acc * batch_len
    return total_acc / data_len


network = Net()
network = network.double()
optimizer = Adam(network.parameters(), lr=0.001, betas=(0.9, 0.999))  # 定义优化器，选用AdamOptimizer
loss_fn = nn.CrossEntropyLoss()  # 定义损失函数，使用交叉熵

# 训练并评估模型

data = Dataset()
model = Model(data)

best_accuracy = 0
for i in range(args.EPOCHS):
    network.train()
    x_train, y_train, x_test, y_test = data.next_batch(args.BATCH)  # 读取数据

    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)

    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)

    optimizer.zero_grad()

    outputs = network(x_train)
    # calculate the loss according to labels
    loss = loss_fn(outputs, y_train)
    # backward transmit loss
    loss.backward()

    _, prediction = torch.max(outputs.data, 1)
    # print(prediction) ##test
    # adjust parameters using Adam
    optimizer.step()

    # 若测试准确率高于当前最高准确率，则保存模型
    train_accuracy = eval(model, x_test, y_test)
    if train_accuracy > best_accuracy:
        best_accuracy = train_accuracy
        model.save_model(network, MODEL_PATH, overwrite=True)
        print("step %d, best accuracy %g" % (i, best_accuracy))

    print(str(i) + "/" + str(args.EPOCHS))
