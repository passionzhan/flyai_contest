# -*- coding: utf-8 -*
import argparse
import math

from numpy import random
from flyai.dataset import Dataset
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler

from model import *
import config
from path import MODEL_PATH
from utilities import data_split
from processor import Processor

# 超参
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=1, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=6, type=int, help="batch size")
args = parser.parse_args()
# 数据获取辅助类
dataset = Dataset(epochs=args.EPOCHS, batch=args.BATCH,val_batch=args.BATCH)
# 模型操作辅助类
model = Model(dataset)

print("number of train examples:%d" % dataset.get_train_length())
print("number of validation examples:%d" % dataset.get_validation_length())

'''
keras: bi-LSTM+CRF
'''


#
# texts = ['中 美 贸 易 战', '中 国 人 民 解 放 军 于 今 日 在 东 海 举 行 实 弹 演 习']
# embeddings = extract_embeddings(pre_trained_path, texts)

ner_model = model.ner_model
ner_model.summary()

x_train, y_train, x_val, y_val = data_split(dataset,val_ratio=0.1)

processor = Processor()
for iLoop in range(len(x_train)):
    y_processed = processor.processXY(x_train[iLoop]['source'], y_train[iLoop]['target'], max_seq_len=config.max_sequence)
    y_train[iLoop]['target'] = y_processed

for iLoop in range(len(x_val)):
    y_processed = processor.processXY(x_val[iLoop]['source'], y_val[iLoop]['target'], max_seq_len=config.max_sequence)
    y_val[iLoop]['target'] = y_processed

# x_train     = dataset.processor_x(x_train)
# x_val       = dataset.processor_x(x_val)
# y_train     = dataset.processor_y(y_train)
# y_val       = dataset.processor_y(y_val)
train_len   = x_train.shape[0]
val_len     = x_val.shape[0]

def gen_batch_data(dataset, x,y,batch_size, max_seq_len=256):
    '''
    批数据生成器
    :param x:
    :param y:
    :param batch_size:
    :return:
    '''
    prcossor = Processor()

    indices = np.arange(x.shape[0])
    random.shuffle(indices)
    x = x[indices]
    y = y[indices]
    i = 0
    while True:
        bi = i*batch_size
        ei = min(i*batch_size + batch_size,len(indices))
        if ei == len(indices):
            i = 0
        else:
            i += 1
        x_batch = x[bi:ei]
        y_batch = y[bi:ei]

        # for iLoop in range(len(x_batch)):
        #     y_processed = prcossor.processXY(x_batch[iLoop]['source'],y_batch[iLoop]['target'],max_seq_len=max_seq_len)
        #     #print(y_processed)
        #     y_batch[iLoop]['target'] = y_processed
        # processor_x   返回的是list 构成的np.array
        x_batch = dataset.processor_x(x_batch)
        y_batch = dataset.processor_y(y_batch)

        x_batch_ids, x_batch_mask, x_batch_seg = conver2Input(x_batch, max_seq_len=max_seq_len)

        y_batch = y_batch.reshape((y_batch.shape[0], y_batch.shape[1], 1))
        yield [x_batch_ids, x_batch_mask], y_batch

steps_per_epoch = math.ceil(train_len / args.BATCH)
val_steps_per_epoch = math.ceil(val_len / args.BATCH)
print("real number of train examples:%d" % train_len)
print("real number of validation examples:%d" % x_val.shape[0])
print("steps_per_epoch:%d" % steps_per_epoch)
print("val_steps_per_epoch:%d" % val_steps_per_epoch)

train_gen   = gen_batch_data(dataset,x_train,y_train,args.BATCH,max_seq_len=config.max_sequence)
val_gen     = gen_batch_data(dataset,x_val,y_val,args.BATCH,max_seq_len=config.max_sequence)


checkpoint = ModelCheckpoint(model.model_path,
                             monitor='val_crf_accuracy',
                             save_best_only=True,
                             save_weights_only=True,
                             verbose=1,
                             mode='max')
earlystop = EarlyStopping(patience=2,)
lrs = LearningRateScheduler(lambda epoch, lr, : 0.9*lr, verbose=1)

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

ner_model.fit_generator(generator=train_gen, steps_per_epoch=steps_per_epoch,
                        epochs=args.EPOCHS,validation_data=val_gen, validation_steps= val_steps_per_epoch,
                        # validation_freq=1,
                        callbacks=[checkpoint, earlystop, lrs])

# # max_val_acc, min_loss = 0, float('inf')
# for i in range(dataset.get_step()):
#     x_train, y_train = dataset.next_train_batch()
#     # padding
#
#     ner_model.train_on_batch(x_train,y_train)
#
#     if i % 50 == 0 or i == dataset.get_step() - 1:
#
#         x_val, y_val = dataset.next_validation_batch()
#         # padding
#         x_val = np.asarray([list(x[:]) + (TIME_STEP - len(x)) * [config.src_padding] for x in x_val])
#         y_val = np.asarray([list(y[:]) + (TIME_STEP - len(y)) * [TAGS_NUM - 1] for y in y_val])
#         y_val = y_train.reshape(y_val.shape[0], y_val.shape[1], 1)
#         val_loss_and_metrics = ner_model.evaluate(x_val, y_val, verbose=0)
#         cur_loss = val_loss_and_metrics[0]
#         cur_acc = val_loss_and_metrics[1]
#
#
#         print('step: %d/%d, val_loss: %f， val_acc: %f'
#               % (i + 1, dataset.get_step(), cur_loss, cur_acc,))
#         # val_loss_and_metrics[1],))
#
#         if max_val_acc < cur_acc \
#                 or (max_val_acc == cur_acc and min_loss > cur_loss):
#             max_val_acc, min_loss = cur_acc, cur_loss
#             print('max_acc: %f, min_loss: %f' % (max_val_acc, min_loss))
#             model.save_model(ner_model, overwrite=True)