import tensorflow as tf
import numpy as np
sess = tf.Session()
x_vals = np.linspace(-10, 10, 1000)
x = tf.placeholder(tf.float32, [1])
print(sess.run(tf.nn.softmax([-1,-2, -0.1, 0.1, 2])))

import requests

birthdata_url = 'https://github.com/nfmcclure/tensorflow_cookbook/raw/master/01_Introduction/07_Working_with_Data_Sources/birthweight_data/birthweight.dat'
birth_file = requests.get(birthdata_url)
print(type(birth_file))
