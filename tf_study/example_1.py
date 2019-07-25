#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : example_1.py
# @Author: Zhan
# @Date  : 7/21/2019
# @Desc  :

import tensorflow as tf

# a = tf.constant([3.0,4.0], dtype=tf.float32)
# b = tf.constant([4.0,5.0]) # also tf.float32 implicitly
#
# c = a + b
#
# writer = tf.summary.FileWriter('.')
# writer.add_graph(tf.get_default_graph())
#
# with tf.Session() as sess:
#     c_rst = sess.run(c)
#     print(c_rst)


# region tensor 学习
# ignition = tf.Variable(451, tf.int16)
# print(ignition)
# sess = tf.compat.v1.Session()
# with sess.as_default():
#     tensor = tf.range(10)
#     print_op = tf.print(tensor)
    # with tf.control_dependencies([print_op]):
    #     out = tf.add(tensor, tensor)
    # sess.run(out)

# v = tf.get_variable("v", shape=(2,), initializer=tf.zeros_initializer())
# u = tf.get_variable('u',shape=(2,),initializer = tf.random_uniform_initializer())
# w = v * u
#
# assign = v.assign_add([2,3])
# with tf.control_dependencies([assign]):
#     ww = v.read_value()
#
mammal = tf.Variable("Elephant", tf.string)
#
#
# first_primes = tf.Variable([2, 3, 5, 7, 11], tf.int32)
#
# index = tf.Variable(2,tf.int32)
# sub_primes = first_primes[index]
# iii = tf.print(sub_primes,)
# with tf.control_dependencies([iii]):
#     rst = sub_primes * sub_primes
#

train_writer = tf.summary.FileWriter('./train')
g_1 = tf.Graph()
with g_1.as_default():

    v = tf.get_variable("v", shape=(2,1), initializer=tf.random_normal_initializer())
    u = tf.get_variable("u",shape=(2,1),initializer=tf.random_normal_initializer())

    # ww = tf.get_variable("u", initializer=tf.zeros_initializer())
    read_v = v.read_value()
    read_u = u.read_value()

    with tf.control_dependencies([read_v,read_u]):
        ww = tf.matmul(v, u, transpose_a=False, transpose_b=True)
        # tf.mat
        # x = ww + u

        train_writer.add_graph(g_1)

test_writer = tf.summary.FileWriter('./test_writer')

g_2 = tf.Graph()
with g_2.as_default():
  # Operations created in this scope will be added to `g_2`.
  x = tf.constant([[37.0, -23.0], [1.0, 4.0]])
  y = tf.Variable(tf.random_uniform([2, 2]))
  z = tf.matmul(x, y)
  test_writer.add_graph(g_2)

with tf.Session(graph=g_1) as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    print(sess.run(read_u))
    print(sess.run(ww))
    print(sess.run(read_u))

    #   # Here we are using the value returned by tf.Print
    # print(sess.run(rst))

    # out = tf.add(index, index)
    # sess.run(out)
    # result = index + 1
#     init = tf.global_variables_initializer()
#     sess.run(init)
#     print(sess.run(sub_primes))
#     print(sess.run(mammal))
# endregion
#
#
#
