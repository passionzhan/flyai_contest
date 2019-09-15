# -*- coding: utf-8 -*
import numpy as np
from flyai.model.base import Base
from keras.layers import Input, Lambda, Dense
from keras_bert import load_trained_model_from_checkpoint
from keras.optimizers import Adam
from keras import Model as kerasModel
from path import *

def conver2Input(x_batch, max_seq_len=256):
    seg_ids         = []
    mask_ids        = []
    input_ids       = []
    #  x 是np array
    for i, x in enumerate(x_batch):
        if len(x) > max_seq_len:
            seg_token = x[-1]
            x = x[0:max_seq_len]
            x[max_seq_len-1] = seg_token
            seg_token_idx = max_seq_len-1
        else:
            seg_token_idx = len(x) - 1
            x = np.concatenate((x, np.asarray([0] * (max_seq_len - len(x)))))

        input_ids.append(list(x))
        tmp_seg = [0] * max_seq_len
        tmp_seg[seg_token_idx] = 1
        tmp_mask = [1] * max_seq_len
        tmp_mask[seg_token_idx+1:] = [0] * (max_seq_len - seg_token_idx - 1)
        seg_ids.append(tmp_seg)
        mask_ids.append(tmp_mask)

    input_ids_batch = np.asarray(input_ids, dtype=np.int32)
    input_mask_batch = np.asarray(mask_ids, dtype=np.int32)
    segment_ids_batch = np.asarray(seg_ids, dtype=np.int32)
    return input_ids_batch, input_mask_batch, segment_ids_batch

class Model(Base):
    def create_model(self):
        # 注意，尽管可以设置seq_len=None，但是仍要保证序列长度不超过512
        bert_model = load_trained_model_from_checkpoint(BERT_CONFIG, BERT_CKPT, seq_len=None)

        for layer in bert_model.layers:
            layer.trainable = True

        x1_in = Input(shape=(None,))
        x2_in = Input(shape=(None,))

        x = bert_model([x1_in, x2_in])
        x = Lambda(lambda x: x[:, 0])(x)  # 取出[CLS]对应的向量用来做分类
        p = Dense(1, activation='sigmoid')(x)

        model = kerasModel([x1_in, x2_in], p)
        model.compile(
            loss='binary_crossentropy',
            optimizer=Adam(1e-5),  # 用足够小的学习率
            metrics=['accuracy']
        )
        # model.summary()
        return model

    def __init__(self, data,):
        self.data = data
        self.model_path = os.path.join(MODEL_PATH, KERAS_BERT_CLASSIFY_MODEL)

        self.spam_model = self.create_model()
        if os.path.isfile(self.model_path):
            print('加载训练好的模型：')
            self.spam_model.load_weights(self.model_path)
            print('加载训练好的模型结束')
        # self.vocab_size = Processor().getWordsCount()

    def predict(self, load_weights = False, **data):
        '''
        预测单条数据
        :param data:
        :return:
        '''
        if load_weights:
            self.spam_model.load_weights(self.model_path)

        x_data = self.data.predict_data(**data)
        input_ids_batch, input_mask_batch, segment_ids_batch = conver2Input(x_data, max_seq_len=256)

        predict = self.spam_model.predict([input_ids_batch, input_mask_batch])
        predict = self.data.to_categorys(predict)[0]
        if predict>=0.5:
            predict = 1
        else:
            predict = 0
        return predict

    def predict_all(self, datas):
        self.spam_model.load_weights(self.model_path)
        self.spam_model.summary()
        predictions = []
        for data in datas:
            prediction = self.predict(**data)
            predictions.append(prediction)
        return predictions

# if __name__ == '__main__':
#
#     # tokenizer = tokenization.FullTokenizer(VOCAB_FILE)
#     # tokens = tokenizer.tokenize("您好！我们这边是施华洛世奇鄞州万达店！您是我们尊贵的会员，特意邀请您参加我们x.x-x.x的三八女人节活动！满xxxx元享晶璨花漾丝巾")
#     # print(tokens)
