# -*- coding: utf-8 -*

from flyai.processor.base import Base
from data_helper import *
import json
import config

class Processor(Base):
    def __init__(self):
        self.token = None
        self.label_dic=config.label_dic
        self.token_dict = tokenizer._token_dict
        self.token_dict_inv = tokenizer._token_dict_inv
        # with open(config.src_vocab_file, 'r') as fw:
        #     self.words_dic = json.load(fw)

    def input_x(self, source):
        '''
        参数为csv中作为输入x的一条数据，该方法会被Dataset多次调用
        '''
        sent_ids = sentence2ids_bert(source,)
        return sent_ids

    def input_y(self, target):
        '''
        参数为csv中作为输入y的一条数据，该方法会被Dataset多次调用
        '''
        label2id = []
        target = target.split()
        for t in target:
            label2id.append(self.label_dic.index(t))
        return label2id

    def output_y(self, index):
        '''
        验证时使用，把模型输出的y转为对应的结果
        '''
        label=[]
        for i in index:
            if i !=config.label_len-1:
                label.append(config.label_dic[i])
            else:
                break
        return label

    def processXY(self, x, y, max_seq_len=128):
        '''
        执行 判决 、 裁定 滥用职权 罪, B-CRIME I-CRIME I-CRIME I-CRIME I-CRIME I-CRIME
        将上述数据转化为：
        执 行 判 决 、 裁 定 滥 用 职 权 罪, B-CRIME I-CRIME I-CRIME I-CRIME I-CRIME I-CRIME I-CRIME I-CRIME I-CRIME I-CRIME I-CRIME I-CRIME
        :param x: 
        :param y: 
        :return: 
        '''
        y = y.split(' ')
        x_list = x.split(' ')
        try:
            assert len(y) == len(x_list)
        except:
            print(y + x_list)
            if len(y) < len(x_list):
                y += ['O',]*(len(x_list) - len(y))
            else:
                y = y[0:len(x_list)]

        # y = y[]
        rst_y = []
        for i, word in enumerate(x_list):
            c_chars = tokenizer.tokenize(word)
            rst_y.append(y[i])
            for j in range(1,len(c_chars)-2):
                if y[i][0] == 'B':
                    rst_y.append('I'+y[i][1:])
                else:
                    rst_y.append(y[i])

        # rst_y.append('O')
        c_chars = tokenizer.tokenize(x)
        assert len(rst_y) + 2 == len(c_chars)

        #   对 Y 进行补齐 #  去掉首位的[cls]
        if len(rst_y) < max_seq_len-1:
            rst_y = rst_y + ['O',]*(max_seq_len-len(rst_y)-1)
        else:
            rst_y = rst_y[0:max_seq_len-1]

        return " ".join(rst_y)

    def processedOutput(self, text, x, y):
        x_char = tokenizer.tokenize(text)
        #以下这种方式导致词典里没有的词还原不回来了，最后导致出错。
        # x_char = [self.token_dict_inv[idx] for idx in x]

        x_char = x_char[1:-1] #去掉首尾
        data_list = text.split(" ")
        yi = 0
        pre_tag = 'O'
        rst_y = []
        for i, word in enumerate(data_list):
            i_word = 0
            try:
                assert word[i_word:len(x_char[yi])+i_word] == x_char[yi]
            except:
                print(word[i_word:len(x_char[yi])+i_word] + "==" + x_char[yi])
            if y[yi][0] == 'B':
                if pre_tag == y[yi]:# 和前一个相同，则改成I
                    rst_y.append('I'+y[yi][1:])
                else:
                    rst_y.append(y[yi])
            else:
                rst_y.append(y[yi])

            pre_tag = rst_y[-1]
            while True:
                i_word += len(x_char[yi])
                yi += 1
                if i_word >= len(word):
                    break
                else:
                    try:
                        assert word[i_word:len(x_char[yi])+i_word] == x_char[yi]
                    except:
                        print("in_loop:" + word[i_word:len(x_char[yi])+i_word] + "==" + x_char[yi])

        if len(rst_y) < len(data_list):
            rst_y += ['O',]*(len(data_list)-len(rst_y))
        else:
            rst_y = rst_y[0:len(data_list)]
        return rst_y


if __name__ == '__main__':
    x = '今年年初 ， 邹士贵 老人 从 箱底 意外 地 翻出 了 一份 他于 1951 年 12 月 1 日 缴纳 人民币 5000 元 （ 旧币 ， 相当于 现在 的 0.5 元 ） 、 投保 20 年期 、 保险金额 为 113 万元 （ 相当于 现在 的 113 元 ） 的 简易 人身 保险单 。'
    y = 'B-TIME O B-PER O O O O O O O O O B-TIME I-TIME I-TIME I-TIME I-TIME I-TIME O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O'.split(' ')

    x, y = Processor().processXY(x,y)
    print(x)
    print(y)











        







