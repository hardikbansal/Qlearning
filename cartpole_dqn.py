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





class dqn():

	def __init__(self, state_size, action_size):

		self.state_size = state_size
		self.action_size = action_size

	def model(self):

		




