import tensorflow as tf
import numpy as np
import sys

from layers import *

import gym


env = gym.make('CartPole-v0')


class network():

	def __init__(self, state_size, action_size, name="network"):

		self.state_size = state_size
		self.action_size = action_size
		self.name = name

	def net(self):

		with tf.variable_scope(self.name) as scope:

			self.input_state = tf.placeholder(tf.float32, [-1, self.state_size], name="input_state")
			weight_mat = tf.Variable(tf.ones(self.state_size, self.action_size),dtype=tf.float32)
			output_weights = tf.nn.sigmoid(tf.matmul(self.input_state, weight_mat))

			self.out_action = tf.argmax(self.output_weights, 1)



class dqn():

	def __init__(self, state_size, action_size):

		self.state_size = state_size
		self.action_size = action_size

	def model(self):

		self.main_net = network(self.state_size, self.action_size, name="main_net")
		self.target_net = network(self.state_size, self.action_size, name="target_net")

		# Initialising the Networks
		self.main_net.net()
		self.target_net.net()

		#Equating the wo networks in the start

	def train(self):

		