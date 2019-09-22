#!/usr/bin/env python 
# encoding: utf-8
import jieba
import numpy as np

from config import load_dict as c_load_dict, max_que_seq_len

def data_clean(text_line):
    text_line = str(text_line)
    if not text_line[0].isalpha():
        text_line = text_line[1:].strip()
    return text_line.strip()


def load_dict():
    return c_load_dict()


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
