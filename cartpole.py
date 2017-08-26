# Simple Policy based agent for Cartpole

# Discounted Rewards are used instead of just current rewards

# For that we need to store the previous actions with weighted rewards

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

		self.state_in = tf.placeholder(dtype=tf.int32, shape=[None, self.state_size])
		temp_states = linear1d(self.state_in, self.state_size, self.action_size)
		self.action = tf.argmax(temp_states)


	def loss_setup(self):

		self.action_hist = tf.placeholder(dtype=tf.int32, shape=[None])
		self.reward_hist = tf.placeholder(dtype=tf.float32, shape=[None])

		self.loss = tf.reduce_mean(tf.loss())




		return 1




	def train(self):

		self.model_setup()
		self.loss_setup()

		init = tf.global_variables_initializer()

		with tf.Session() as sess:

			sess.run(init)

			for i in range(episodes):

				for j in range(max_iter):

					temp = np.random.uniform(1)

					if temp > e :
						temp_action = sess.run(self.action, feed_dict={self.state_in:curr_state})
					else :
						temp_action = np.random.randint(2, size=[1])

					news_state, reward, done, _ = env.step(temp_action[0])

					if(done):
						break

def main():

	model = contextual_bandits()
	model.train()

if __name__ == "__main__":
	main()