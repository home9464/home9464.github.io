import tensorflow as tf
import numpy as np
import os
import sys
import math
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#tf.executing_eagerly() 
def add_timing_signal_nd(x, min_timescale=1.0, max_timescale=1.0e4):
    """Adds a bunch of sinusoids of different frequencies to a Tensor.
    Each channel of the input Tensor is incremented by a sinusoid of a difft
    frequency and phase in one of the positional dimensions.
    This allows attention to learn to use absolute and relative positions.
    Timing signals should be added to some precursors of both the query and the
    memory inputs to attention.
    The use of relative position is possible because sin(a+b) and cos(a+b) can
    be experessed in terms of b, sin(a) and cos(a).
    x is a Tensor with n "positional" dimensions, e.g. one dimension for a
    sequence or two dimensions for an image
    We use a geometric sequence of timescales starting with
    min_timescale and ending with max_timescale.  The number of different
    timescales is equal to channels // (n * 2). For each timescale, we
    generate the two sinusoidal signals sin(timestep/timescale) and
    cos(timestep/timescale).  All of these sinusoids are concatenated in
    the channels dimension.
    Args:
        x: a Tensor with shape [batch, d1 ... dn, channels]
        min_timescale: a float
        max_timescale: a float
    Returns:
        a Tensor the same shape as x.
    """
    static_shape = x.get_shape().as_list()  # 2,3,4
    num_dims = len(static_shape) - 2  # 1 which is 3, without batch size and channel size
    channels = tf.shape(x)[-1]  # 4 channels
    num_timescales = channels // (num_dims * 2)  # 2 scales
    print('num_timescales: {}'.format(num_timescales))
    # log(2, 10000) / 
    log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            (tf.to_float(num_timescales) - 1))
    #inv_timescales: 1 * [0, 1, ...] * time_scale_unit -> [1, 0.000099]
    inv_timescales = min_timescale * tf.exp(
            tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    for dim in range(num_dims):  # just one dimension
        length = tf.shape(x)[dim + 1]# add 1 to skip batch size
        position = tf.to_float(tf.range(length)) # say, 3 positions of [0, 1, 2]
        #  scaled_time = [[0],[1],[2]] * [[1, 0.000099]]
        scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(
                inv_timescales, 0)
        # PE(pos, 2i) = sin(pos/ 10000^(2i/d_{model}))
        # PE(pos, 2i+1) = sin(pos/ 10000^(2i/d_{model})), where i is channel

        # side by side, merge from horizontally
        signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
        prepad = dim * 2 * num_timescales  #0*2*2,  1*2*2, 2*2*2 ...
        postpad = channels - (dim + 1) * 2 * num_timescales
        signal = tf.pad(signal, [[0, 0], [prepad, postpad]])
        for _ in range(1 + dim):
            signal = tf.expand_dims(signal, 0)
        for _ in range(num_dims - 1 - dim):
            signal = tf.expand_dims(signal, -2)  # why?
        x += signal
        print('signal: {}'.format(signal))
        print('x: {}'.format(x))
    return x


# for NLP, the dim is [batch_size, sequence_length, embedding_size, 1]

tf.enable_eager_execution()
x = tf.convert_to_tensor(np.arange(2*3*3*10).reshape(2, 3, 3, 10), dtype=tf.float32)
y = add_timing_signal_nd(x, min_timescale=1.0, max_timescale=1.0e4)
#with tf.Session() as sess:
#    print(sess.run(y))
