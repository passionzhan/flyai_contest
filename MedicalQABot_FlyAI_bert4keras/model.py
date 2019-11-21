# -*- coding: utf-8 -*

import numpy as np
from flyai.model.base import Base
from keras.optimizers import Adam
import jieba
from keras import backend as K
from nltk.translate.bleu_score import sentence_bleu

from bert4keras.bert import build_bert_model

from data_helper import myToken
from config import *
from utilities import label_smoothing

def bleu_score(y_output, y_true):
    answer = y_true[0]
    y_mask = y_true[1]
    token_rst = np.argmax(y_output,axis=-1)
    score = 0.
    for i in range(y_output.shape[0]):
        token_rst_list = list(token_rst[i][y_mask[i]==1])
        cur_pred = myToken.get_tokenizer().decode(token_rst_list)
        cur_pred = jieba.lcut(cur_pred)
        cur_ans = jieba.lcut(answer[i])
        print("当前预测的结构：%s" % cur_pred)
        print("正确答案：%s" % cur_ans)
        score += sentence_bleu(cur_pred,cur_ans)
    return score/y_output.shape[0]

def create_model():
    bert4nlg_model = build_bert_model(BERT_CONFIG,
         checkpoint_path=BERT_CKPT,
         with_mlm=False,
         application='seq2seq',
         keep_words=myToken.get_token_dict()[1],
         albert=False,
         return_keras_model=True)

    # bert4nlg_model.summary()

    # 交叉熵作为loss，并mask掉输入部分的预测
    #  由于输入中，没有传入实际的 Y 值，所以各种准确度度量不起作用
    y_in = bert4nlg_model.input[0][:, 1:]  # 目标tokens
    y_mask = bert4nlg_model.input[1][:, 1:]
    y = bert4nlg_model.output[:, :-1]  # 预测tokens，预测与目标错开一位
    Y_in_one_hot = K.one_hot(K.cast(y_in,dtype="int32"), len(myToken.get_token_dict()[0]))
    y_in_smoothing = label_smoothing(Y_in_one_hot,epsilon=0.3)
    cross_entropy = K.categorical_crossentropy(y_in_smoothing, y)
    cross_entropy = K.sum(cross_entropy * y_mask) / K.sum(y_mask)

    bert4nlg_model.add_loss(cross_entropy)
    bert4nlg_model.compile(optimizer=Adam(learning_rate),metrics=[bleu_score,])

    return bert4nlg_model

class Model(Base):
    def __init__(self, data):
        self.data = data
        self.model_path = os.path.join(MODEL_PATH, QA_MODEL_DIR)
        self.bert4nlg_model = create_model()
        if os.path.isfile(self.model_path):
            print('加载训练好的模型：')
            self.bert4nlg_model.load_weights(self.model_path)
            print('加载训练好的模型结束')

    # 基于beam reseach的解码
    def decode_sequence(self, input_seq, max_decode_seq_length=max_ans_seq_len_predict,topk=2):
        tokenizer = myToken.get_tokenizer()
        token_dict = tokenizer._token_dict
        IGNORE_WORD_IDX = token_dict[FIRST_VALIDED_TOKEN]

        token_ids, segment_ids = tokenizer.encode(input_seq[:max_que_seq_len-2])
        input_seq = np.tile(token_ids,(topk,1))
        input_seg = np.tile(segment_ids,(topk,1))

        pre_score   = np.array([[0.]*topk]*topk)

        stop_condition = False
        target_seq = None
        target_seq_output = None
        while not stop_condition:
            output_tokens = self.bert4nlg_model.predict([input_seq,input_seg])[:,-1,IGNORE_WORD_IDX:]

            arg_topk = output_tokens.argsort(axis=-1)[:, -topk:]  # 每一项选出topk

            #首次输出，三个样本一样，所以取第一个样本topk就行
            if target_seq is None:
                target_seq = arg_topk[0, :].reshape((topk,1)) + IGNORE_WORD_IDX
                tmp_cur_score = np.log(np.sort(output_tokens[0,:], axis=-1)[-topk:])
                tmp_cur_score = np.tile(tmp_cur_score.reshape((topk,1)),(1,topk))
                cur_score = pre_score + tmp_cur_score
                pre_score = cur_score
                target_seq_output = target_seq

            else:
                # 当上次输出中有结束符 ‘[SEP]’ 时，将该样本输出'[SEP]' _sentence_end_token的概率置为最大1.0
                for i, word in enumerate(target_seq[:, 0]):
                    if word == token_dict[sentence_end_token]:
                        output_tokens[i, token_dict[sentence_end_token] - IGNORE_WORD_IDX] = 1.0
                # pre_score
                # 取对数防止向下溢出
                # 利用对数计算，乘法改+法
                tmp_cur_score = np.log(np.sort(output_tokens, axis=-1)[:, -topk:])
                cur_score = pre_score + tmp_cur_score
                maxIdx = np.unravel_index(np.argsort(cur_score, axis=None)[-topk:],cur_score.shape)
                pre_score = np.tile(cur_score[maxIdx].reshape((topk,1)),(1,topk))
                target_seq  = arg_topk[maxIdx].reshape((topk,1)) + IGNORE_WORD_IDX

                target_seq_output = np.concatenate((target_seq_output[maxIdx[0],:],target_seq),axis=-1)

            if (target_seq_output.shape[1] >= max_decode_seq_length
                    or (target_seq == token_dict[sentence_end_token] * np.ones((topk,1))).all()):
                stop_condition = True

            targt_seg   = np.array([[1]] * topk,dtype=input_seg.dtype)
            input_seq = np.concatenate((input_seq,target_seq),axis=-1)
            input_seg = np.concatenate((input_seg,targt_seg),axis=-1)

        # print("==")
        # 最后一行，概率最大
        # maxIdx为元组，维数为 pre_score维度值
        # maxIdx = np.unravel_index(np.argmax(pre_score, axis=None), pre_score.shape)
        # print(maxIdx[0])
        target_seq_output = target_seq_output[-1,:].reshape(1,-1)
        for i, word in enumerate(target_seq_output[0,:]):
            if word == token_dict[sentence_end_token]:
                break
        target_seq_output = target_seq_output[:,0:i]
        return target_seq_output

    def predict(self, load_weights = False, **data):
        '''

        :param data:
        :return:
        '''

        x_data = data["que_text"]

        predict = self.decode_sequence(x_data,max_decode_seq_length=max_ans_seq_len_predict)
        predict = myToken.get_tokenizer().decode(predict[0])
        return predict

    def predict_all(self, datas):
        self.bert4nlg_model.load_weights(self.model_path)
        predictions = []
        for data in datas:
            prediction = self.predict(**data)
            predictions.append(prediction)
        return predictions


