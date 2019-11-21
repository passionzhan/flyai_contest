#!/usr/bin/env python 
# encoding: utf-8
import codecs

import jieba
import numpy as np
from flyai.dataset import Dataset
from bert4keras.utils import Tokenizer, load_vocab
from path import *
from config import *
from tqdm import tqdm

def load_myvocab(dataset):
    if os.path.exists(MY_VOCAB_FILE):
        chars = json.load(open(MY_VOCAB_FILE, encoding='utf-8'))
    else:
        chars = {}
        x_train, y_train, x_val, y_val = dataset.get_all_data()
        x_data = np.concatenate((x_train, x_val))
        y_data = np.concatenate((y_train, y_val))

        for q in tqdm(x_data, desc=u'构建字表中_处理问题'):
            for w in q["que_text"]:  # 纯文本，不用分词
                chars[w] = chars.get(w, 0) + 1
        for a in tqdm(y_data, desc=u'构建字表中_处理回答'):
            for w in a["ans_text"]:  # 纯文本，不用分词
                chars[w] = chars.get(w, 0) + 1

        chars = [(char, count) for char, count in chars.items() if count >= min_count]
        chars = sorted(chars, key=lambda c: - c[1])
        chars = [c[0] for c in chars]
        json.dump(
            chars,
            codecs.open(MY_VOCAB_FILE, 'w', encoding='utf-8'),
            indent=4,
            ensure_ascii=False
        )

    _token_dict = load_vocab(VOCAB_FILE)  # 读取词典
    token_dict, keep_words = {}, []

    for c in ['[PAD]', '[UNK]', '[CLS]', '[unused1]', '[SEP]']:
        token_dict[c] = len(token_dict)
        keep_words.append(_token_dict[c])

    for c in chars:
        if c in _token_dict:
            token_dict[c] = len(token_dict)
            keep_words.append(_token_dict[c])

    return token_dict, keep_words

class myToken():
    dataset = None
    token_dict = None
    keep_words = None
    tokenizer = None  # 建立分词器

    @classmethod
    def get_dataset(cls,):
        if cls.dataset is not None:
            return cls.dataset
        else:
            cls.dataset = Dataset(epochs=3, batch=3, val_batch=3)
            return cls.dataset

    @classmethod
    def get_token_dict(cls,):
        if (cls.token_dict is not None) and (cls.keep_words is not None):
            return cls.token_dict, cls.keep_words
        else:
            cls.token_dict, cls.keep_words = load_myvocab(cls.get_dataset())
            return cls.token_dict, cls.keep_words

    @classmethod
    def get_tokenizer(cls,):
        if cls.tokenizer is not None:
            return cls.tokenizer
        else:
            cls.tokenizer = Tokenizer(cls.get_token_dict()[0])
            return cls.tokenizer

def data_clean(text_line):
    text_line = str(text_line)
    if not text_line[0].isalpha():
        text_line = text_line[1:].strip()
    return text_line.strip()


def id2ans(ans_list, ans_dict):
    id2ans = dict()
    for key, value in ans_dict.items():
        id2ans[str(value)] = key
    tmp_list = list()
    for it in ans_list:
        tmp_list.append(id2ans[str(it)])
    return tmp_list


def que_process(que_line, que_dict, max_que_len=max_que_seq_len):
    que_line = jieba.lcut(que_line)
    que_len = len(que_line)
    que_list = list()
    for i in range(len(que_line)):
        if que_line[i] in que_dict.keys():
            que_list.append(que_dict[que_line[i]])
        else:
            que_list.append(que_dict['_unk_'])

    if len(que_list) < max_que_len:
        que_list += [que_dict['_pad_'] for _ in range(max_que_len - len(que_list))]
    else:
        que_list = que_list[:max_que_len]
    return que_list, que_len


def ans_process(ans_line, ans_dict):
    ans_line = jieba.lcut(ans_line)
    ans_list = list()
    for i in range(len(ans_line)):
        if ans_line[i] in ans_dict.keys():
            ans_list.append(ans_dict[ans_line[i]])
        else:
            ans_list.append(ans_dict['_unk_'])
    ans_list.append(ans_dict['_eos_'])
    ans_len = len(ans_list)
    return ans_list, ans_len


def process_ans_batch(ans_batch, ans_dict, max_ans_len):
    ans_list = list()
    for line in ans_batch:
        line = list(line)
        if len(line) < max_ans_len:
            line += [ans_dict['_pad_'] for _ in range(max_ans_len - len(line))]
        else:
            line = line[:max_ans_len]
        ans_list.append(line)
    ans_array = np.asarray(ans_list)
    return ans_array


if __name__ == "__main__":
    exit(0)
