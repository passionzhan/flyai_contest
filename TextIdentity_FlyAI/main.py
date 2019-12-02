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
from model import Model, get_NumofGPU, getDevive
from path import MODEL_PATH
from transformers.optimization import AdamW
from transformers import AlbertTokenizer
from config import *
from utilities import data_split

def padding(x, pad_token):
    """padding至batch内的最大长度
    """
    ml = max([len(i) for i in x])
    padded_x = [sequence +  [pad_token] * (ml - len(sequence)) for sequence in x]
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
    tokenizer = AlbertTokenizer(ALBERT_PATH)

    while True:
        bi = i*batch_size
        ei = min(i*batch_size + batch_size,len(indices))
        if ei == len(indices):
            i = 0
        else:
            i += 1

        x_train = [tokenizer.encode(text["usr_text"][0:max_que_seq_len-2],text_pair=text['ans_comment'],max_length = max_seq_len) for text in x[bi:ei]]
        y_train = [label['label'] for label in y[bi:ei]]

        x_train = padding(x_train,tokenizer.pad_token_id)

        x_train = torch.tensor(x_train,dtype=torch.long)
        y_train = torch.tensor(y_train,dtype=torch.long)

        yield x_train,y_train

'''
项目的超参
'''
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=10, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=32, type=int, help="batch size")
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

test_x_data = x_val[-args.BATCH:]
test_y_data = y_val[-args.BATCH:]


steps_per_epoch = math.ceil(train_len / args.BATCH)
val_steps_per_epoch = math.ceil(val_len / args.BATCH)
print("real number of train examples:%d" % train_len)
print("real number of validation examples:%d" % x_val.shape[0])
print("steps_per_epoch:%d" % steps_per_epoch)
print("val_steps_per_epoch:%d" % val_steps_per_epoch)

train_gen   = gen_batch_data(x_train,y_train,args.BATCH)
val_gen     = gen_batch_data(x_val, y_val,args.BATCH)

if not os.path.exists(mymodel.model_path):
    os.makedirs(mymodel.model_path)

optimizer  = AdamW(TI_net.parameters(),lr=learning_rate)
TI_net.zero_grad()

epoch_iterator      = trange(args.EPOCHS, desc="Epoch", disable=True)
step_iterator       = trange(steps_per_epoch, desc="STEP", disable=True)
step_iterator_val   = trange(val_steps_per_epoch, desc="VAL_STEP", disable=True)
global_step = 0
tr_loss = 0.0
minloss_val = float('inf')
best_score = 0

for epoch in epoch_iterator:
    for step in step_iterator:
        '''
        准备数据,训练
        '''
        x_train, y_train = next(train_gen)
        TI_net.train()
        outputs = TI_net(x_train, labels=y_train)
        loss, logits = outputs[:2]
        loss.backward()
        optimizer.step()
        TI_net.zero_grad()

    if (step+1) % VAL_FREQUENCY == 0 or (step+1) == steps_per_epoch:
        '''
            准备数据，评估
        '''

        for val_step in step_iterator_val:
            x_val, y_val = next(val_gen)
            TI_net.eval()
            outputs = TI_net(x_val,labels=y_val)
            loss, logits = outputs[:2]
            '''
            实现自己的模型保存逻辑
            '''
            mymodel.save_model()
        print(str(step + 1) + "/" + str(dataset.get_step()))