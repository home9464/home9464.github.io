import tensorflow as tf
import numpy as np
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

"""
out = tf.layers.conv2d(inputs=img,
                       filters=64,  # number of filters per filter size
                       kernel_size=3,  # height and width of conv window
                       strides=1,  # steps on height and width
                       padding='same',  # add padding around the input
                       activation=tf.nn.relu)
out = tf.layers.max_pooling2d(inputs=out,
                              pool_size=2,  # (pooling height, pooling width)
                              strides=2,  # (strides height, strides width)
                              padding='same')  

out = tf.layers.conv2d(inputs=img,
                       filters=128,  # number of filters per filter size
                       kernel_size=3,  # height and width of conv window
                       strides=1,  # steps on height and width
                       padding='same',  # add padding around the input
                       activation=tf.nn.relu)
out = tf.layers.max_pooling2d(inputs=out,
                              pool_size=2,  # (pooling height, pooling width)
                              strides=2,  # (strides height, strides width)
                              padding='same')  


out = tf.layers.conv2d(inputs=img,
                       filters=256,  # number of filters per filter size
                       kernel_size=3,  # height and width of conv window
                       strides=1,  # steps on height and width
                       padding='same',  # add padding around the input
                       activation=tf.nn.relu)
out = tf.layers.max_pooling2d(inputs=out,
                              pool_size=(2, 1),  # (pooling height, pooling width)
                              strides=(2, 1)  # (strides height, strides width)
                              padding='same')

out = tf.layers.conv2d(inputs=img,
                       filters=512,  # number of filters per filter size
                       kernel_size=3,  # height and width of conv window
                       strides=1,  # steps on height and width
                       padding='same',  # add padding around the input
                       activation=tf.nn.relu)
out = tf.layers.max_pooling2d(inputs=out,
                              pool_size=(1, 2),  # (pooling height, pooling width)
                              strides=(1, 2)  # (strides height, strides width)
                              padding='same')


"""
