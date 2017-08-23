# Simple Policy based agent for Cartpole

import tensorflow as tf
import numpy as np

import gym


env = gym.make('CartPole-v0')


class cartpole():

	def __init__(self):

		self.state = 0


	def model_setup(self):

		return 1


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