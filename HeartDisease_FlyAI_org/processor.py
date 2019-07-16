# -*- coding: utf-8 -*
import numpy
from flyai.processor.base import Base


def Normalize(data):
    m = numpy.mean(data)
    mx = max(data)
    mn = min(data)
    return [(float(i) - m) / (mx - mn) for i in data]


class Processor(Base):

    def input_x(self, age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
        x_data = numpy.zeros(13)  ## 输入维度为13
        x_data[:] = age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
        x_data = Normalize(x_data)

        return x_data

    def input_y(self, label):
        label -= 1  ##从（1，2） 变为（0，1）
        return label

    def output_y(self, data):
        return numpy.argmax(data) + 1
