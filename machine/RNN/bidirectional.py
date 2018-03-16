import tensorflow as tf
import numpy as np
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


# https://www.quora.com/How-is-the-hidden-state-h-different-from-the-memory-c-in-an-LSTM-cell

# num_units is dimensiona of the hidden state and output state
# if state_is_tuple=True then state = [cell_state, hidden_state]

def forward():
    """
    outputs: The RNN output Tensor.

        If time_major == False (default), this will be a Tensor 
        shaped: [batch_size, max_time, cell.output_size].

        If time_major == True, this will be a Tensor 
        shaped: [max_time, batch_size, cell.output_size].

        Note, if cell.output_size is a (possibly nested) tuple of integers 
        or TensorShape objects, then outputs will be a tuple having the 
        same structure as cell.output_size, containing Tensors having 
        shapes corresponding to the shape data in cell.output_size.

    state: The final state. If cell.state_size is an int, 
        this will be shaped [batch_size, cell.state_size]. 
        If it is a TensorShape, 
        this will be shaped [batch_size] + cell.state_size. 
        If it is a (possibly nested) tuple of ints or TensorShape, 
        this will be a tuple having the corresponding shapes. 
        If cells are LSTMCells state will be a tuple containing a 
        LSTMStateTuple for each cell.

    """
    tf.reset_default_graph()
    input_x = tf.placeholder(shape=[2, 5, 6], dtype=tf.float32)
    cell = tf.nn.rnn_cell.LSTMCell(num_units=16, state_is_tuple=True)
    outputs, state = tf.nn.dynamic_rnn(
        cell=cell,
        dtype=tf.float32,
        #sequence_length=[5, 3],  # length (# rows) for two examples
        time_major=False,
        inputs=input_x)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        outputs_, state_ = sess.run([outputs, state], feed_dict={input_x: x})
        # 'outputs_' is a tensor of shape [batch_size, max_time, cell.output_size]
        print(outputs_.shape)  # array with shape: (2, 5, 16)

        print(type(state_))  # it is the state, LSTMStateTuple, comes from "state_is_tuple=True"
        print(state_[0].shape)  # (batch_size, cell.hidden_size), the cell state
        print(state_[1].shape)  # (batch_size, cell.hidden_size), the hidden state
                



def bidirectional():
    tf.reset_default_graph()
    input_x = tf.placeholder(shape=[2, 5, 6], dtype=tf.float32)
    cell = tf.nn.rnn_cell.LSTMCell(num_units=16, state_is_tuple=True)
    outputs, states = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=cell,
        cell_bw=cell,
        dtype=tf.float32,
        sequence_length=[5, 3],  # length (# rows) for two examples
        time_major=False,
        inputs=input_x)
    output_fw, output_bw = outputs
    states_fw, states_bw = states
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run([outputs, states], feed_dict={input_x: x})



# out shape: [2, 5, num_units*2]
#out = tf.concat(outputs, 2)  # on 3rd dim

# out shape: [2, 2, 32]
#state = tf.concat(states, 2)  # on 3rd dim
#state = states_fw  # on 3rd dim

# shape: [2, 5, 6], [batch, ]
x = np.random.randn(2, 5, 6)
# row 4 and row5 in the second example are all 0
#x[1, 3:] = 0

#bidirectional()
forward()