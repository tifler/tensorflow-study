# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 10:18:40 2017

@author: tifler
"""

import tensorflow as tf
import numpy as np


x_data = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]
y_data = [[0.], [1.], [1.], [1.]]

x_data = np.array(x_data, dtype=np.float32)
y_data = np.array(y_data, dtype=np.float32)

X = tf.placeholder(dtype=tf.float32, shape=[None, 2], name='X')
Y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='Y')

W = tf.Variable(tf.random_normal(shape=[2, 1]), name='weight')
b = tf.Variable(tf.random_normal(shape=[1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(10001):
        v_c, v_W, v_b, _ = sess.run([cost, W, b, train], feed_dict={X: x_data, Y: y_data})
        if step % 1000 == 0:
            print(step, v_c, v_W, v_b)
            
    h = sess.run(hypothesis, feed_dict={X: x_data, Y: y_data})
    print (h)
        
        
