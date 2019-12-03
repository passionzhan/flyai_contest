# -*- coding: utf-8 -*-
import argparse
import math
import functools
import logging

import numpy as np
from numpy import random
from flyai.dataset import Dataset
import jieba
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm, trange
import torch
from torch.optim import Adam
from transformers import AutoTokenizer

from model import Model, getDevive
from utilities import data_split
from config import *
from utils_summarization import compute_token_type_ids, build_mask, build_lm_labels

class BertSumOptimizer(object):
    """ Specific optimizer for BertSum.

    As described in [1], the authors fine-tune BertSum for abstractive
    summarization using two Adam Optimizers with different warm-up steps and
    learning rate. They also use a custom learning rate scheduler.

    [1] Liu, Yang, and Mirella Lapata. "Text summarization with pretrained encoders."
        arXiv preprint arXiv:1908.08345 (2019).
    """

    def __init__(self, model, lr, warmup_steps, beta_1=0.99, beta_2=0.999, eps=1e-8):
        self.encoder = model.encoder
        self.decoder = model.decoder
        self.lr = lr
        self.warmup_steps = warmup_steps

        self.optimizers = {
            "encoder": Adam(
                model.encoder.parameters(),
                lr=lr["encoder"],
                betas=(beta_1, beta_2),
                eps=eps,
            ),
            "decoder": Adam(
                model.decoder.parameters(),
                lr=lr["decoder"],
                betas=(beta_1, beta_2),
                eps=eps,
            ),
        }

        self._step = 0

    def _update_rate(self, stack):
        return self.lr[stack] * min(
            self._step ** (-0.5), self._step * self.warmup_steps[stack] ** (-0.5)
        )

    def zero_grad(self):
        self.optimizer_decoder.zero_grad()
        self.optimizer_encoder.zero_grad()

    def step(self):
        self._step += 1
        for stack, optimizer in self.optimizers.items():
            new_rate = self._update_rate(stack)
            for param_group in optimizer.param_groups:
                param_group["lr"] = new_rate
            optimizer.step()

'''
项目中的超参
'''
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=3, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=3, type=int, help="batch size")
parser.add_argument("-vb", "--VAL_BATCH", default=3, type=int, help="val batch size")
args = parser.parse_args()
device, n_gpu = getDevive()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

#  在本样例中， args.BATCH 和 args.VAL_BATCH 大小需要一致
'''
flyai库中的提供的数据处理方法
传入整个数据训练多少轮，每批次批大小
'''
dataset = Dataset(epochs=args.EPOCHS, batch=args.BATCH, val_batch=args.VAL_BATCH)
mymodel = Model(dataset)

print("number of train examples:%d" % dataset.get_train_length())
print("number of validation examples:%d" % dataset.get_validation_length())

QA_model = mymodel.QA_model
print(QA_model)

x_train, y_train, x_val, y_val = data_split(dataset,val_ratio=0.1)
train_len   = x_train.shape[0]
val_len     = x_val.shape[0]

test_x_data = x_val[-args.BATCH:]
test_y_data = y_val[-args.BATCH:]


def show_result(model, que_data, ans_data):
    score = 0.0
    for i, que in enumerate(que_data):
        predict = model.predict(**que)
        print("预测结果：%s"%predict)
        print("实际答案：%s" % ans_data[i]["ans_text"])
        score += sentence_bleu([jieba.lcut(predict)], jieba.lcut(ans_data[i]["ans_text"]), weights=(1., 0., 0., 0))

    print("当前bleu得分：%f" % (score / que_data.shape[0]))

def padding(x,pad_token):
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

    tokenizer = AutoTokenizer.from_pretrained(BERT_PATH)
    indices = np.arange(x.shape[0])
    random.shuffle(indices)
    x = x[indices]
    y = y[indices]
    i = 0

    x_batch, y_batch, answer = [], [], []
    while True:
        bi = i*batch_size
        ei = min(i*batch_size + batch_size,len(indices))
        if ei == len(indices):
            i = 0
        else:
            i += 1

        # for idx in range(bi,ei):
        #     # 确保编码后也不超过max_seq_len
        #     x_      = x[idx]["que_text"][:max_que_seq_len-3]
        #     y_      = y[idx]["ans_text"][:max_ans_seq_len]
        #     # 加入答案主要是为了评估进行模型选择用
        #     #answer.append(y_)
        #     x_, y_ = myToken.get_tokenizer().encode(x_, y_)
        #     x_batch.append(x_)
        #     y_batch.append(y_)

        # x_batch = padding(x_batch)
        # y_batch = padding(y_batch)
        #answer  = np.array(answer)
        # yield [x_batch, y_batch], None
        # tokenizer = AutoTokenizer.from_pretrained(BERT_PATH)
        # source, target, encoder_token_type_ids, encoder_mask, decoder_mask, lm_labels = batch
        # x_batch, y_batch, answer = [], [], []

        # data = filter(lambda x: not (len(x[0]) == 0 or len(x[1]) == 0), data)
        data_que = [tokenizer.encode(que["que_text"][0:max_que_seq_len-2]) for que in x[bi:ei]]
        data_ans = [tokenizer.encode(ans["ans_text"][0:max_ans_seq_len-2]) for ans in y[bi:ei]]

        data_que = padding(data_que,tokenizer.pad_token_id)
        data_ans = padding(data_ans, tokenizer.pad_token_id)

        ques = torch.tensor(data_que,dtype=torch.long)
        anss = torch.tensor(data_ans,dtype=torch.long)
        encoder_token_type_ids = compute_token_type_ids(ques, tokenizer.sep_token_id)
        encoder_mask = build_mask(ques, tokenizer.pad_token_id)
        decoder_mask = build_mask(anss, tokenizer.pad_token_id)
        lm_labels = build_lm_labels(anss, tokenizer.pad_token_id)

        yield (
            ques,
            anss,
            encoder_token_type_ids,
            encoder_mask,
            decoder_mask,
            lm_labels,
        )

steps_per_epoch = math.ceil(train_len / args.BATCH)
val_steps_per_epoch = math.ceil(val_len / args.BATCH)
print("real number of train examples:%d" % train_len)
print("real number of validation examples:%d" % x_val.shape[0])
print("steps_per_epoch:%d" % steps_per_epoch)
print("val_steps_per_epoch:%d" % val_steps_per_epoch)

train_gen   = functools.partial(gen_batch_data, x_train,y_train,args.BATCH)()
val_gen     = functools.partial(gen_batch_data, x_val, y_val,args.BATCH)()

if not os.path.exists(mymodel.model_path):
    os.makedirs(mymodel.model_path)

if not os.path.exists(os.path.join(mymodel.model_path,"encoder")):
    os.makedirs(os.path.join(mymodel.model_path, "encoder"))

if not os.path.exists(os.path.join(mymodel.model_path,"decoder")):
    os.makedirs(os.path.join(mymodel.model_path,"decoder"))



# 超参数
batch_size = args.BATCH

# Prepare the optimizer
lr = {"encoder": 0.0005, "decoder": 0.2}
warmup_steps = {"encoder": 20000, "decoder": 10000}
optimizer = BertSumOptimizer(QA_model, lr, warmup_steps)
#  训练
QA_model.zero_grad()
epoch_iterator = trange(args.EPOCHS, desc="Epoch", disable=True)

global_step = 0
tr_loss = 0.0
minloss_val = float('inf')
for epoch in epoch_iterator:
    step_iterator = trange(steps_per_epoch, desc="Step", disable=True)
    for step in step_iterator:
        questions, answers, question_token_type_ids, question_mask, answer_mask, ans_lm_labels = next(train_gen)

        questions = questions.to(device)
        answers = answers.to(device)
        question_token_type_ids = question_token_type_ids.to(device)
        question_mask = question_mask.to(device)
        answer_mask = answer_mask.to(device)
        ans_lm_labels = ans_lm_labels.to(device)

        QA_model.train()
        outputs = QA_model(
            questions,
            answers,
            encoder_token_type_ids=question_token_type_ids,
            encoder_attention_mask=question_mask,
            decoder_attention_mask=answer_mask,
            decoder_lm_labels=ans_lm_labels,
        )

        loss = outputs[0]
        loss.backward()

        torch.nn.utils.clip_grad_norm_(QA_model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        QA_model.zero_grad()

        if (step+1) % VAL_FREQUENCY == 0 or (step+1) == steps_per_epoch:
            #  展示效果
            show_result(mymodel,test_x_data,test_y_data)
            loss_val = 0
            QA_model.eval()
            for i in range(VAL_STEPS_PER_VAL):
                que_val, ans_val, que_token_type_ids_val, que_mask_val, ans_mask_val, ans_lm_labels_val = next(val_gen)

                que_val = que_val.to(device)
                ans_val = ans_val.to(device)
                que_token_type_ids_val = que_token_type_ids_val.to(device)
                que_mask_val = que_mask_val.to(device)
                ans_mask_val = ans_mask_val.to(device)
                ans_lm_labels_val = ans_lm_labels_val.to(device)

                outputs = QA_model(
                    que_val,
                    ans_val,
                    encoder_token_type_ids=que_token_type_ids_val,
                    encoder_attention_mask=que_mask_val,
                    decoder_attention_mask=ans_mask_val,
                    decoder_lm_labels=ans_lm_labels_val,
                )
                loss_val += outputs[0].item()

            loss_val = loss_val/VAL_STEPS_PER_VAL
            logger.info(
                "step:{0}/epoch:{1}, train_loss:{2}, val_loss:{3}".format(step, epoch, loss.item(), loss_val))

            if loss_val <= minloss_val:
                logger.info("min loss improved from {0} to {1}, save the model!".format(minloss_val,loss_val))
                minloss_val = loss_val
                mymodel.save()