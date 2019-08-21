# -*- coding: utf-8 -*
import os
from flyai.model.base import Base
import tensorflow as tf
import tensorflow.keras.backend as K
# import keras.backend as K
# import tensorflow.python.keras.backend as K
from path import MODEL_PATH

TF_MODEL_NAME = "densenet.ckpt"
class Model(Base):
    def __init__(self, dataset):
        self.dataset = dataset
        self.model_path = os.path.join(MODEL_PATH, TF_MODEL_NAME)

    '''
    评估一条数据
    '''

    def predict(self, sess, x_input, pred_y, **prediction):
        # 获取需要预测的图像数据， predict_data 方法默认会去调用 processor.py 中的 input_x 方法
        x_data = self.dataset.predict_data(**prediction)
        predict = sess.run(pred_y, feed_dict={x_input: x_data, K.learning_phase(): 0})
        # 将预测数据转换成对应标签  to_categorys 会去调用 processor.py 中的 output_y 方法
        prediction = self.dataset.to_categorys(predict)
        return prediction

    '''
    评估的时候会调用该方法实现评估得分
    '''

    def predict_all(self, datas):
        with tf.Session() as sess:
            pred_y, x_inputs = self.loadmodel(sess)

            predictions = []
            for data in datas:
                prediction = self.predict(sess, x_inputs, pred_y, **data)
                # 将预测数据转换成对应标签  to_categorys 会去调用 processor.py 中的 output_y 方法
                data = self.dataset.to_categorys(prediction)
                predictions.append(data)
        return predictions

    def loadmodel(self, sess):
        ckpt = tf.train.get_checkpoint_state(MODEL_PATH)
        meta_file = ckpt.model_checkpoint_path + '.meta'
        # 加载图
        saver = tf.train.import_meta_graph(meta_file)
        # 加载参数
        saver.restore(sess, ckpt.model_checkpoint_path)
        graph = tf.get_default_graph()
        x_inputs = graph.get_tensor_by_name("x_inputs:0")
        pred_y = graph.get_tensor_by_name("pred_y:0")
        return pred_y, x_inputs

    '''
    保存模型的方法
    '''
    def save_model(self, sess, path, name=TF_MODEL_NAME, overwrite=False):
        super().save_model(sess, path, name, overwrite)
        saver = tf.train.Saver(var_list=tf.global_variables())
        saver.save(sess, os.path.join(path, name))

    # def save_model(self, sess, path, name=TF_MODEL_NAME, overwrite=False):

