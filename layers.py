import tensorflow as tf
import numpy as np



def linear1d(inputlin, inputdim, outputdim, name="linear1d", std=0.02, use_bias=True):

    with tf.variable_scope(name) as scope:

        weight = tf.Variable(tf.truncated_normal([inputdim, outputdim], stddev=0.01), name="weight")
        bias = tf.Variable(tf.zeros(outputdim), name='bias')

        return tf.matmul(inputlin, weight) + bias