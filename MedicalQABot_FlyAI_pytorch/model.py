# -*- coding: utf-8 -*

import numpy as np
from flyai.model.base import Base
import jieba
import torch
from transformers import (
    AutoTokenizer,
    BertForMaskedLM,
    BertConfig,
    PreTrainedEncoderDecoder,
    Model2Model,
)

from nltk.translate.bleu_score import sentence_bleu
from utils_summarization import compute_token_type_ids, build_mask

from config import *
from path import BERT_PATH

def getDevive():# Set up training device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cpu")
        n_gpu = 0
    return device, n_gpu

def fit_to_block_size(sequence, block_size, pad_token):
    """ Adapt the source and target sequences' lengths to the block size.
    If the sequence is shorter than the block size we pad it with -1 ids
    which correspond to padding tokens.
    """
    if len(sequence) > block_size:
        return sequence[:block_size]
    else:
        sequence.extend([pad_token] * (block_size - len(sequence)))
        return sequence

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
    config = BertConfig.from_pretrained(BERT_PATH)
    decoder_model = BertForMaskedLM(config)
    QA_model = Model2Model.from_pretrained(BERT_PATH, decoder_model=decoder_model)

    return QA_model

class Model(Base):
    def __init__(self,dataset):
        self.dataset = dataset
        self.model_path = os.path.join(MODEL_PATH,QA_MODEL_DIR)
        if os.path.isdir(self.model_path):
            print('加载训练好的模型：')
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            encoder_path = os.path.join(self.model_path, "encoder")
            decoder_path = os.path.join(self.model_path, "decoder")
            self.QA_model = PreTrainedEncoderDecoder.from_pretrained(
                encoder_path, decoder_path
            )
            print('加载训练好的模型结束')
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(BERT_PATH)
            self.QA_model = create_model()

    # 基于beam reseach的解码
    def decode_sequence(self, encoder_seq, max_decode_seq_length=max_ans_seq_len_predict,topk=2):
        tokenizer = self.tokenizer
        token_dict = tokenizer.vocab
        IGNORE_WORD_IDX = token_dict[FIRST_VALIDED_TOKEN]
        encoder_token_ids = [tokenizer.encode(encoder_seq[:max_que_seq_len-2])]
        # 扩充为 topk个样本
        encoder_token_ids = torch.tensor(encoder_token_ids).repeat((topk,1)).to(getDevive()[0])
        encoder_token_type_ids = compute_token_type_ids(encoder_token_ids, tokenizer.sep_token_id)
        encoder_mask = build_mask(encoder_token_ids, tokenizer.pad_token_id,)

        # 编码
        self.QA_model.eval()
        encoder_outputs = self.QA_model.encoder(encoder_token_ids, token_type_ids= encoder_token_type_ids,attention_mask = encoder_mask)
        encoder_hidden_states = encoder_outputs[0]

        # 解码
        input_seq = torch.tensor([[tokenizer.cls_token_id,]]).repeat((topk,1)).to(getDevive()[0])
        pre_score = np.array([[0.]*topk]*topk)


        stop_condition = False
        target_seq = None
        target_seq_output = None
        while not stop_condition:
            # 准备数据
            decoder_mask = build_mask(input_seq, tokenizer.pad_token_id).to(getDevive()[0])
            #  预测时无法计算标签损失，所以不提供 lm_labels
            outputs = self.QA_model.decoder(input_seq,
                                            encoder_hidden_states=encoder_hidden_states,
                                            encoder_attention_mask=encoder_mask,
                                            attention_mask=decoder_mask,
            )
            output_tokens = outputs[0][:,-1,IGNORE_WORD_IDX:].data.numpy()

            #首次输出，三个样本一样，所以取第一个样本topk就行
            if target_seq is None:
                arg_topk = output_tokens.argsort(axis=-1)[:, -topk:]  # 每一项选出topk
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
                arg_topk = output_tokens.argsort(axis=-1)[:, -topk:]  # 每一项选出topk
                tmp_cur_score = np.log(arg_topk)
                cur_score = pre_score + tmp_cur_score
                maxIdx = np.unravel_index(np.argsort(cur_score, axis=None)[-topk:],cur_score.shape)
                pre_score = np.tile(cur_score[maxIdx].reshape((topk,1)),(1,topk))
                target_seq  = arg_topk[maxIdx].reshape((topk,1)) + IGNORE_WORD_IDX

                target_seq_output = np.concatenate((target_seq_output[maxIdx[0],:],target_seq),axis=-1)

            if (target_seq_output.shape[1] >= max_decode_seq_length
                    or (target_seq == token_dict[sentence_end_token] * np.ones((topk,1))).all()):
                stop_condition = True

            input_seq = torch.cat((input_seq,torch.from_numpy(target_seq)),axis=-1)

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

    def predict(self, **data):
        '''
        :param data:
        :return:
        '''
        x_data = data["que_text"]
        predict = self.decode_sequence(x_data,max_decode_seq_length=max_ans_seq_len_predict)
        predict = self.tokenizer.decode(predict[0])
        return predict

    def predict_all(self, datas):
        predictions = []
        for data in datas:
            prediction = self.predict(**data)
            predictions.append(prediction)
        return predictions

    def save(self):
        model_to_save = (
            self.QA_model.module if hasattr(self.QA_model, "module") else self.QA_model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(self.model_path)
        self.tokenizer.save_pretrained(self.model_path)


