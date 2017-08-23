import tensorflow as tf
import numpy as np



def linear1d(inputlin, inputdim, outputdim, name="linear1d", std=0.02, use_bias="True"):

    with tf.variable_scope(name) as scope:

        weight = tf.get_variable("weight",[inputdim, outputdim])
        bias = tf.get_variable("bias",[outputdim], dtype=np.float32, initializer=tf.constant_initializer(0.0))

        if use_bias:
            return tf.matmul(inputlin, weight) + bias
        else:
            return tf.matmul(inputlin, weight)