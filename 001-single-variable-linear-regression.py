
import tensorflow as tf
import numpy as np

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

#W = tf.Variable(tf.random_normal([1]), name='weight')
#b = tf.Variable(tf.random_normal([1]), name='bias')
W = tf.Variable(5.0)
b = tf.Variable(10.0)

hypothesis = X * W + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    v_cost, v_W, v_b, _ = sess.run([cost, W, b, train], feed_dict={X: [1, 2, 3], Y: [2, 3, 4]})
    if step % 20 == 0:
        print(step, v_cost, v_W, v_b)

# test result
h = sess.run(hypothesis, feed_dict={X: [3]})
print("h(3) = ", h)
