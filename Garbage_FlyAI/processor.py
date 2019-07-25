# -*- coding: utf-8 -*
import cv2
import numpy
from flyai.processor.base import Base
from flyai.processor.download import check_download

from path import DATA_PATH

'''
把样例项目中的processor.py件复制过来替换即可
'''


class Processor(Base):
    '''
    参数为csv中作为输入x的一条数据，该方法会被dataset.next_train_batch()
    和dataset.next_validation_batch()多次调用。可在该方法中做数据增强
    该方法字段与app.yaml中的input:->columns:对应
    '''

    def input_x(self, image_path):
        path = check_download(image_path, DATA_PATH)
        image = cv2.imread(path)
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
        x_data = numpy.array(image)
        x_data = x_data.astype(numpy.float32)
        x_data = numpy.multiply(x_data, 1.0 / 255.0)
        return x_data

    '''
    参数为csv中作为输入y的一条数据，该方法会被dataset.next_train_batch()
    和dataset.next_validation_batch()多次调用。
    该方法字段与app.yaml中的output:->columns:对应
    '''

    def output_x(self, image_path):
        path = check_download(image_path, DATA_PATH)
        image = cv2.imread(path)
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
        x_data = numpy.array(image)
        x_data = x_data.astype(numpy.float32)
        x_data = numpy.multiply(x_data, 1.0 / 255.0)
        return x_data

    '''
    参数为csv中作为输入x的一条数据，该方法会被dataset.next_train_batch()
    和dataset.next_validation_batch()多次调用。评估的时候会调用该方法做数据处理
    该方法字段与app.yaml中的input:->columns:对应
    '''

    def input_y(self, labels):
        one_hot_label = numpy.zeros([6])  ##生成全0矩阵
        one_hot_label[labels] = 1  ##相应标签位置置
        return one_hot_label

    '''
    输出的结果，会被dataset.to_categorys(data)调用
    '''

    def output_y(self, data):
        return numpy.argmax(data)
