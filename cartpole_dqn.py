import tensorflow as tf
import numpy as np
import sys
import random

from layers import *

import gym

env = gym.make('CartPole-v0')


class network():

	def __init__(self, state_size, action_size, h1_size=10, h2_size = 5, name="network"):

		self.state_size = state_size
		self.action_size = action_size
		self.name = name
		self.h1_size = h1_size
		self.h2_size = h2_size

	def net(self):

		with tf.variable_scope(self.name) as scope:

			self.input_state = tf.placeholder(tf.float32, [None, self.state_size], name="input_state")

			h1 = linear1d(self.input_state, self.state_size, self.h1_size, name="hidden1")
			h2 = linear1d(h1, self.h1_size, self.h2_size, name="hidden2")

			self.output_weights = tf.nn.sigmoid(linear1d(h2, self.h2_size, self.action_size))

			weight_action = tf.Variable(tf.ones([self.state_size, self.action_size]), name="weight_action")

			self.out_action = tf.argmax(self.output_weights, 1)

class dqn():

	def __init__(self, state_size, action_size):

		# Defining the hyper parameters

		self.state_size = state_size
		self.action_size = action_size
		self.eps = 0.1

		self.num_episodes = 1000
		self.max_steps = 200
		self.pre_train_steps = 20
		self.update_freq = 10
		self.batch_size = 20
		self.gamma = 0.99
		self.lr = 0.001

	def copy_network(self, net1, net2):

		vars1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, net1.name)
		vars2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, net2.name)

		for idx in range(len(vars1)):
			vars2[idx].assign(vars1[idx])

	def model(self):

		self.main_net = network(self.state_size, self.action_size, name="main_net")
		self.target_net = network(self.state_size, self.action_size, name="target_net")

		# Initialising the Networks
		self.main_net.net()
		self.target_net.net()

		# Defining the model for the training

		self.target_reward = tf.placeholder(tf.float32, [None, 1], name="target_reward")
		self.action_list = tf.placeholder(tf.int32, [None, 1], name="action_list")

		observed_reward = tf.reduce_sum(self.main_net.output_weights*tf.one_hot(self.action_list,2,dtype=tf.float32),1)

		self.loss = tf.reduce_sum(tf.square(self.target_reward - observed_reward))

		optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
		self.loss_opt = optimizer.minimize(self.loss)


	def train(self):

		self.model()
		init = tf.global_variables_initializer()


		with tf.Session() as sess:

			sess.run(init)
			self.copy_network(self.main_net, self.target_net)
			
			total_steps = 0

			for i in range(self.num_episodes):

				curr_state = env.reset()

				for j in range(1, self.max_steps+1):

					temp = np.random.random()
					
					if temp < self.eps or total_steps < self.pre_train_steps:
						temp_action = np.random.randint(self.action_size, size=[1])
					else :
						temp_action = sess.run([self.main_net.out_action[0]], feed_dict={self.main_net.input_state:np.reshape(curr_state,[-1, self.state_size])})

					# print(temp_action.shape)
					new_state, reward, done, _ = env.step(temp_action[0])


					if(total_steps == 0):
						hist_buffer = np.array([[temp_action, curr_state, new_state, reward, done]])
					else :
						hist_buffer = np.insert(hist_buffer, hist_buffer.shape[0], np.array([temp_action, curr_state, new_state, reward, done]), axis=0)

					if (done):
						break

					if(total_steps > self.pre_train_steps):

						if(total_steps % self.update_freq == 0):

							rand_batch = hist_buffer[np.random.choice(hist_buffer.shape[0], self.batch_size, replace=False)]

							reward_hist = rand_batch[:,3]
							state_hist = rand_batch[:,1]
							action_hist = np.vstack(rand_batch[:,0])
							next_state_hist = rand_batch[:,2]

							temp_target_q = sess.run(self.main_net.output_weights, feed_dict={self.main_net.input_state:np.vstack(next_state_hist)})

							temp_target_q = np.amax(temp_target_q,1)
							temp_target_reward = reward_hist + self.gamma*temp_target_q
							temp_target_reward =  np.reshape(temp_target_reward, [self.batch_size, 1])
							
							# print(action_hist.shape)


							_ = sess.run(self.loss_opt, feed_dict={self.main_net.input_state:np.vstack(state_hist), self.target_reward:temp_target_reward, self.action_list:action_hist})

							self.copy_network(self.main_net, self.target_net)
						
							sys.exit()


					# if(total_steps == 0):

					curr_state = new_state

					total_steps+=1

def main():

	model = dqn(4,2)
	model.train()

if __name__ == "__main__":
	main()