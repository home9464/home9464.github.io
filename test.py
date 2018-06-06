import os
import tensorflow as tf
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

batch_size = 1
image_height = 4
image_width = 4
image_channels = 1  # 3 for RGB or 1 for greyscale

x_shape = [batch_size, image_height, image_width, image_channels]
#image_x = np.random.uniform(size=x_shape)
image_x = np.arange(1, 17).reshape(x_shape)
#image_x = np.expand_dims(image_x, 0)
#image_x = np.expand_dims(image_x, -1)
#print(image_x.shape)
#print(image_x)

x_data = tf.placeholder(shape=x_shape, dtype=tf.float32)

filter_height = 2
filter_width = 2
filter_in_channels = 1
filter_out_channels = 1

filter_shape = [filter_height, filter_width, filter_in_channels, filter_out_channels]
my_filter = tf.constant(0.25, shape=filter_shape)

stride_size_on_rows = 1
stride_size_on_cols = 2
my_stride = [1, stride_size_on_rows, stride_size_on_cols, 1]
moving_avg_layer = tf.nn.conv2d(input=x_data, filter=my_filter, strides=my_stride, padding='SAME')


sess = tf.Session()
ret = sess.run(moving_avg_layer, feed_dict={x_data: image_x})
print(ret)
print(ret.shape)
#x_vals = np.linspace(-10, 10, 1000)
#x = tf.placeholder(tf.float32, [1])
#print(sess.run(tf.nn.softmax([-1,-2, -0.1, 0.1, 2])))
#print(sess.run(tf.nn.moments(tf.constant([-1,-2, -0.1, 0.1, 2]), axes=[0])))
#print(sess.run(tf.random_normal(shape=[2, 10], mean=10, stddev=1)))
#print(sess.run(tf.mod(22.4, 5.0)))


