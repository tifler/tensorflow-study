# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 12:13:55 2017

@author: tifler
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


x_data = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]], dtype=np.float32)
y_data = np.array([[0.], [1.], [1.], [0.]], dtype=np.float32)

X = tf.placeholder(dtype=tf.float32, shape=[None, 2], name='X')
Y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='Y')

with tf.name_scope("layer1") as scope:
    W1 = tf.Variable(tf.random_normal(shape=[2, 8]), name='weight1')
    b1 = tf.Variable(tf.random_normal(shape=[8]), name='bias1')
    layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)
    
with tf.name_scope("layer2") as scope:
    W2 = tf.Variable(tf.random_normal(shape=[8, 1]), name='weight2')
    b2 = tf.Variable(tf.random_normal(shape=[1]), name='bias2')
    hypothesis = tf.sigmoid(tf.matmul(layer1, W2) + b2)

with tf.name_scope("cost") as scope:    
    #cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
    cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
    
with tf.name_scope("train") as scope:
    #optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    #train = optimizer.minimize(cost)
    train = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

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
    print ("Hypothesis:", h, "Predicted:", p, "Accuracy:", a)        
        