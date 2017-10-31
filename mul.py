# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 16:43:43 2017

@author: tifler
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

raw_x = [
        [1., 1.], [2., 1.],[2.,2.],[3., 3.], [10.,10.]
        ]
raw_y = [
        [1.], [2.],[4.], [9.], [100.]
        ]

x_data = np.array(raw_x, dtype=np.float32)
y_data = np.array(raw_y, dtype=np.float32)

X = tf.placeholder(dtype=tf.float32, shape=[None, 2], name='X')
Y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='Y')

with tf.name_scope("layer1") as scope:
    W1 = tf.Variable(tf.random_normal(shape=[2, 32]), name='weight1')
    b1 = tf.Variable(tf.random_normal(shape=[32]), name='bias1')
    layer1 = tf.matmul(X, W1) + b1
    
with tf.name_scope("layer2") as scope:
    W2 = tf.Variable(tf.random_normal(shape=[32, 1]), name='weight2')
    b2 = tf.Variable(tf.random_normal(shape=[1]), name='bias2')
    hypothesis = tf.matmul(layer1, W2) + b2

with tf.name_scope("cost") as scope:    
    cost = tf.reduce_mean(tf.square(hypothesis - Y))
    
with tf.name_scope("train") as scope:
    #optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    #train = optimizer.minimize(cost)
    train = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

predicted = tf.cast(hypothesis, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.abs(predicted-Y))

cost_array = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(10001):
        v_c, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
        if step % 100 == 0:
            print(step, v_c)
            cost_array.append(v_c)
            
    
    plt.plot(cost_array, label="cost")
    plt.legend()
    plt.show()

    h, p, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data})
    print ("\nPredicted:", p, "\nY:", y_data, "\nAccuracy:", a)        

    tx = np.array([[11., 11.], [5., 3.]])
    ty = np.array([[121.], [15.]])
    h, p, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: tx, Y: ty})
    print ("\nPredicted:", p, "\nY:", ty, "\nAccuracy:", a)        
        
