import tensorflow as tf
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


class Seq2Seq(object):
    def __init__(self,
                 xseq_len,  # encoder sequence
                 yseq_len,  # decoder sequence
                 xvocab_size,
                 yvocab_size,
                 emb_dim,  # embedding dimension
                 num_layers,
                 ckpt_path,  # checkpoint path
                 lr=0.0001
                 epochs=100000,
                 model_name='seq2seq_model'
                 )
        """
        Args:
            xseq_len encoder sequence
            yseq_len decoder sequence
            xvocab_size
            yvocab_size
            emb_dim embedding dimension
            num_layers
            ckpt_path checkpoint path
        
        """
        self.xseq_len = xseq_len
        self.yseq_len = yseq_len
        self.ckpt_path = ckpt_path
        self.epochs = epochs
        self.model_name = model_name

        def graph():
            tf.reset_default_graph()
            # encoder inputs
            self.enc_ip = [tf.placeholder(shape=[None, ],
                dtype=tf.int64, name='ei+{}'.format(t)) for t in range(xseq_len)]
            # labels of true outputs
            self.labels = [tf.placeholder(shape=[None, ],
                dtype=tf.int64, name="ei_{}".format(t)) for t in range(yseq_len)]
            # decoder inputs: "GO"+ [y1, y2, ... y_t-1]
            self.dec_ip = [tf.zero_like(self.enc_ip[0], dtype=tf.int64, name='GO')] + self.labels[:-1]
            
