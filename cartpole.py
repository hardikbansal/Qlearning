# Simple Policy based agent for Cartpole

import tensorflow as tf
import numpy as np

from layers import *

import gym


env = gym.make('CartPole-v0')


class cartpole():

	def __init__(self):

		self.state = 0
		self.state_size = 4
		self.action_size = 2


	def model_setup(self):

		self.state_in = tf.placeholder(dtype=tf.int32, shape=[self.state_size])

		temp_states = linear1d(self.state_in, self.state_size, self.action_size)

		state_out = tf.argmax(temp_states)



	def loss_setup(self):

		return 1




	def train(self):

		self.model_setup()
		self.loss_setup()

		init = tf.global_variables_initializer()

		with tf.Session() as sess:


def main():

	model = contextual_bandits()
	model.train()

if __name__ == "__main__":
	main()