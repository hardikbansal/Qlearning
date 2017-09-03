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
			weight_action = tf.Variable(tf.ones(self.state_size, self.action_size),dtype=tf.float32, name="weight_action")
			weight_reward = tf.Variable(tf.ones(self.state_size, 1),dtype=tf.float32, name="weight_reward")
			weight_done = tf.Variable(tf.ones(self.state_size, 1),dtype=tf.float32, name="weight_done")
			output_weights = tf.nn.sigmoid(tf.matmul(self.input_state, weight_action))

			self.out_action = tf.argmax(self.output_weights, 1)
			self.out_reward = tf.matmul(self.input_state, weight_reward)
			self.out_done = tf.matmul(self.input_state, weight_done)



class dqn():

	def __init__(self, state_size, action_size):

		self.state_size = state_size
		self.action_size = action_size
		self.eps = 0.1

	def copy_network(self, net1, net2):

		vars1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, net1.name)
		vars2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, net2.name)

		for idx in range(len(vars1)):
			vars2[idx].assign(val).eval()

	def model(self):

		self.main_net = network(self.state_size, self.action_size, name="main_net")
		self.target_net = network(self.state_size, self.action_size, name="target_net")

		# Initialising the Networks
		self.main_net.net()
		self.target_net.net()

		#Equating the wo networks in the start

		copy_network(self.main_net, self.target_net)


	def train(self):

		self.model()
		init = tf.global_variables_initializer()

		s = env.reset()

		with tf.Session() as sess:

			for i in range(num_episodes):

				temp = np.random.random()
				
				if(temp > self.eps):
					