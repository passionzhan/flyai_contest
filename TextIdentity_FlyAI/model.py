# -*- coding: utf-8 -*
import os
import tensorflow as tf
from flyai.model.base import Base
from path import MODEL_PATH
from tensorflow.python.saved_model import tag_constants

TENSORFLOW_MODEL_DIR = "best"


class Model(Base):
    def __init__(self, data):
        self.data = data
        self.model_path = os.path.join(MODEL_PATH, TENSORFLOW_MODEL_DIR)
        self.session = tf.Session()
        self.is_load = False
        if os.path.exists(self.model_path):
            tf.saved_model.loader.load(self.session, [tag_constants.SERVING], self.model_path)
            self.is_load = True

    def predict(self, **data):
        '''
        使用模型
        :param path: 模型所在的路径
        :param name: 模型的名字
        :param data: 模型的输入参数
        :return:
        '''
        if not self.is_load:
            tf.saved_model.loader.load(self.session, [tag_constants.SERVING], self.model_path)
            self.is_load = True
        input_x = self.session.graph.get_tensor_by_name(self.get_tensor_name('input_x'))
        input_y = self.session.graph.get_tensor_by_name(self.get_tensor_name('input_y'))
        keep_prob = self.session.graph.get_tensor_by_name(self.get_tensor_name('keep_prob'))

        x_data = self.data.predict_data(**data)
        predict = self.session.run(input_y, feed_dict={input_x: x_data, keep_prob: 1.0})
        return self.data.to_categorys(predict)

    def predict_all(self, datas):
        if not self.is_load:
            tf.saved_model.loader.load(self.session, [tag_constants.SERVING], self.model_path)
            self.is_load = True
        input_x = self.session.graph.get_tensor_by_name(self.get_tensor_name('input_x'))
        input_y = self.session.graph.get_tensor_by_name(self.get_tensor_name('input_y'))
        keep_prob = self.session.graph.get_tensor_by_name(self.get_tensor_name('keep_prob'))

        labels = []
        for data in datas:
            x_data = self.data.predict_data(**data)
            predict = self.session.run(input_y, feed_dict={input_x: x_data, keep_prob: 1.0})
            label = self.data.to_categorys(predict)
            labels.append(label)
        return labels

    def save_model(self, session, path, name=TENSORFLOW_MODEL_DIR, overwrite=False):
        '''
        保存模型
        :param session: 训练模型的sessopm
        :param path: 要保存模型的路径
        :param name: 要保存模型的名字
        :param overwrite: 是否覆盖当前模型
        :return:
        '''
        if overwrite:
            self.delete_file(path)

        builder = tf.saved_model.builder.SavedModelBuilder(os.path.join(path, name))
        builder.add_meta_graph_and_variables(session, [tf.saved_model.tag_constants.SERVING])
        builder.save()

    def get_tensor_name(self, name):
        return name + ":0"

    def delete_file(self, path):
        for root, dirs, files in os.walk(path, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))