import tensorflow as tf
import numpy as np
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

#x = tf.constant([[1,1], [2,2]], dtype=tf.int32)
sess = tf.Session()
#print(sess.run(tf.shape(x)[1:2]))
#x = np.array([1,2,3,4,5])
#y = np.random.choice(x, x.shape[0], replace=False)
#print(y)
#print(np.dot(x, y))
#print(sess.run(tf.nn.softmax([3.,2.,1.]/np.sqrt(3))))
#print(sess.run(tf.nn.softmax([6.,4.,2.]/np.sqrt(3))))
a = np.array([3.0, -4., -1.])
b = np.array([0., 5., 2.])
a = np.array([2, 1., -1.])
b = np.array([1., 0., -2.])
magnitude_a = np.sqrt(np.dot(a, a))
magnitude_b = np.sqrt(np.dot(b, b))
cos_ab = a.dot(b) / (magnitude_a*magnitude_b)
angle_ab = np.arccos(cos_ab) * 180 / 3.1415
#print(angle_ab)

projection_of_b_on_a = a.dot(b) * a / np.dot(a, a)
projection_of_a_on_b = a.dot(b) * b / np.dot(b, b)
print(projection_of_b_on_a)
print(projection_of_a_on_b)



