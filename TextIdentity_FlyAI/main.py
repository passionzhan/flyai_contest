# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 19:44:02 2017

@author:
"""
import os
import argparse
import math

import numpy as np
from numpy import random
from tqdm import trange
import torch
from flyai.dataset import Dataset
from model import Model, getDevive
from transformers.optimization import AdamW
from transformers import BertTokenizer

# 导入flyai打印日志函数的库
from flyai.utils.log_helper import train_log

from config import *
from utilities import data_split

def padding(x, pad_token):
    """padding至batch内的最大长度
    """
    ml = max([len(i) for i in x])
    padded_x = [sequence + [pad_token] * (ml - len(sequence)) for sequence in x]
    return padded_x

def gen_batch_data(x,y, batch_size):
    '''
    批数据生成器
    :param x:
    :param y:
    :param batch_size:
    :return:
    '''
    indices = np.arange(x.shape[0])
    random.shuffle(indices)
    x = x[indices]
    y = y[indices]
    i = 0
    tokenizer = BertTokenizer(os.path.join(ALBERT_PATH, "vocab.txt"))

    while True:
        bi = i*batch_size
        ei = min(i*batch_size + batch_size,len(indices))
        if ei == len(indices):
            i = 0
        else:
            i += 1

        x_train = [tokenizer.encode(text["usr_text"],text_pair=text['ans_comment'],max_length = max_seq_len) for text in x[bi:ei]]
        y_train = [label['label'] for label in y[bi:ei]]

        x_train = padding(x_train,tokenizer.pad_token_id)

        x_train = torch.tensor(x_train,dtype=torch.long).to(getDevive())
        y_train = torch.tensor(y_train,dtype=torch.long).to(getDevive())
        # x_train = tf.constant(x_train, dtype=torch.long)
        # y_train = tf.constant(y_train, dtype=torch.long)
        yield x_train,y_train

'''
项目的超参
'''
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=20, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=3, type=int, help="batch size")
args = parser.parse_args()

'''
flyai库中的提供的数据处理方法
传入整个数据训练多少轮，每批次批大小
'''
dataset = Dataset(epochs=args.EPOCHS, batch=args.BATCH)
mymodel = Model(dataset)

print("number of train examples:%d" % dataset.get_train_length())
print("number of validation examples:%d" % dataset.get_validation_length())

TI_net = mymodel.net
print(TI_net)

x_train, y_train, x_val, y_val = data_split(dataset,val_ratio=0.1)
train_len   = x_train.shape[0]
val_len     = x_val.shape[0]

steps_per_epoch = math.ceil(train_len / args.BATCH)
val_steps_per_epoch = math.ceil(val_len / args.BATCH)
print("real number of train examples:%d" % train_len)
print("real number of validation examples:%d" % x_val.shape[0])
print("steps_per_epoch:%d" % steps_per_epoch)
print("val_steps_per_epoch:%d" % val_steps_per_epoch)

train_gen   = gen_batch_data(x_train,y_train,args.BATCH)
val_gen     = gen_batch_data(x_val, y_val,args.BATCH)

if not os.path.exists(mymodel.net_path):
    os.makedirs(mymodel.net_path)

optimizer = AdamW(TI_net.parameters(),lr=learning_rate,weight_decay=0.01)
TI_net.zero_grad()

epoch_iterator      = trange(args.EPOCHS, desc="Epoch", disable=True)
step_iterator       = trange(steps_per_epoch, desc="STEP", disable=True)
val_step_iterator   = trange(VAL_STEPS_PER_VAL, desc="STEPS_VAL", disable=True)
max_acc = 0.

for epoch in epoch_iterator:
    for step in step_iterator:
        '''
        准备数据,训练
        '''
        x_train, y_train = next(train_gen)
        TI_net.train()
        outputs = TI_net(x_train, labels=y_train)
        loss, logits = outputs[:2]
        TI_net.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss = loss.item()
        print("step " + str(step + 1) + "/" + str(steps_per_epoch) + ", " +
                "epoch " + str(epoch + 1) + "/" + str(args.EPOCHS) + ", " +
                "train loss: " + str(train_loss))

        if (step+1) % VAL_FREQUENCY == 0 or (step+1) == steps_per_epoch:
            '''
            准备数据，评估
            '''
            val_loss = 0.0
            right_num = 0
            n_smp = 0
            for val_step in val_step_iterator:
                x_val, y_val = next(val_gen)
                n_smp += y_val.size()[0]
                TI_net.eval()
                outputs = TI_net(x_val,labels=y_val)
                val_loss +=outputs[0].item()
                logits = outputs[1]
                pred = logits.argmax(dim=-1)
                right_num +=torch.eq(pred,y_val).sum().item()
                '''
                实现自己的模型保存逻辑
                '''
            val_loss =val_loss / (val_step + 1)
            val_acc = right_num / n_smp
            print("step " + str(step + 1) + "/" + str(steps_per_epoch) + ", " +
                  "epoch " + str(epoch + 1) + "/" + str(args.EPOCHS) + ", "
                  + "val loss is " + str(val_loss) + ", val acc is " + str(val_acc))
            # 调用系统打印日志函数，这样在线上可看到训练和校验准确率和损失的实时变化曲线
            train_log(train_loss=train_loss, train_acc=0.5, val_loss=val_loss, val_acc=val_acc)
            if max_acc < val_acc:
                print("acc improved from {0} to {1}, model saved.".format(max_acc, val_acc))
                max_acc = val_acc
                mymodel.save_model()



