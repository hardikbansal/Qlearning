# Simple Policy based agent for Cartpole

# Discounted Rewards are used instead of just current rewards

# For that we need to store the previous actions with weighted rewards

import tensorflow as tf
import numpy as np
import sys

from layers import *

import gym


env = gym.make('CartPole-v0')


class cartpole():

	def __init__(self):

		self.state = 0

		# State size os 4 correspodong tp theta, accn, distance, velocity
		self.state_size = 4

		# An action denote the force of magnitude 1 in the left or right direction corresponding to 0/1
		self.action_size = 2
		self.max_iter = 10
		self.num_episodes = 1
		self.e = 0.1


	def model_setup(self):

		# Here we suppose that the network is a fully connected layer

		self.state_in = tf.placeholder(dtype=tf.int32, shape=[None, self.state_size])
		self.prob_action = linear1d(tf.cast(self.state_in, tf.float32), self.state_size, self.action_size)
		
		# Taking the action based on the weights

		self.action = tf.argmax(self.prob_action, 1)


	def loss_setup(self):

		action_hist = tf.placeholder(dtype=tf.int32, shape=[None,self.action_size])
		reward_hist = tf.placeholder(dtype=tf.float32, shape=[None])

		# Caclulation the temp weight by taking the weights corresponding action that we got earlier in the stage

		temp_weights = tf.reduce_sum(self.prob_action*tf.cast(action_hist, tf.float32), 1)

		self.loss = tf.reduce_mean(-tf.log(temp_weights)*reward_hist)

		# Calculating the gradients in tensorflow




	def train(self):

		self.model_setup()
		self.loss_setup()

		init = tf.global_variables_initializer()

		with tf.Session() as sess:

			sess.run(init)

			for i in range(self.num_episodes):

				curr_state = env.reset()

				for j in range(self.max_iter):

					temp = np.random.uniform(1)

					if temp > self.e :
						temp_action = sess.run(self.action, feed_dict={self.state_in:np.reshape(curr_state,[-1, self.state_size])})
					else :
						temp_action = np.random.randint(self.action_size, size=[1])

					new_state, reward, done, _ = env.step(temp_action[0])

					if (j == 0):
						history = np.array([[temp_action, curr_state, new_state, reward]])
					else:
						history = np.insert(history, history.shape[0], np.array([temp_action, curr_state, new_state, reward]), axis=0)

					temp_grad = sess.run(self.gradients, feed_dict={self.state_in:np.reshape(history[:,2],[-1, self.state_size])})


					print(history.shape)

					# sys.exit()

					# Here I am applying the gradients after some fixed number of steps.


					curr_state = new_state

					if(done):
						break

def main():

	model = cartpole()
	model.train()

if __name__ == "__main__":
	main()