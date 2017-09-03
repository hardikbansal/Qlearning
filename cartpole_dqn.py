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

			self.input_state = tf.placeholder(tf.float32, [None, self.state_size], name="input_state")
			weight_action = tf.Variable(tf.ones([self.state_size, self.action_size]), name="weight_action")
			weight_reward = tf.Variable(tf.ones([self.state_size, 1]), name="weight_reward")
			weight_done = tf.Variable(tf.ones([self.state_size, 1]), name="weight_done")
			output_weights = tf.nn.sigmoid(tf.matmul(self.input_state, weight_action))

			self.out_action = tf.argmax(output_weights, 1)
			self.out_done = tf.matmul(self.input_state, weight_done)



class dqn():

	def __init__(self, state_size, action_size):

		self.state_size = state_size
		self.action_size = action_size
		self.eps = 0.1

		self.num_episodes = 1000
		self.max_steps = 200
		self.pre_train_steps = 1000
		self.batch_size = 20

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

		observed_reward = self.main_net.output_weights*tf.one_hot(self.main_net.out_action)

		self.loss = tf.reduce_sum(tf.square(self.target_reward - observed_reward))

		optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
		optimizer.minimize(self.loss, lr=0.0001)


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
						temp_action = sess.run([self.main_net.out_action], feed_dict={self.main_net.input_state:np.reshape(curr_state,[-1, self.state_size])})

					new_state, reward, done, _ = env.step(temp_action[0])

					if(total_steps == 0):
						hist_buffer = np.array([[temp_action, curr_state, new_state, reward, done]])
					else :
						hist_buffer = np.insert(hist_buffer, hist_buffer.shape[0], np.array([temp_action, curr_state, new_state, reward, done]), axis=0)

					# print(hist_buffer)


					if(total_steps > self.pre_train_steps):

						if(total_steps % self.update_freq == 0):

							rand_batch = random.sample(hist_buffer, self.batch_size)

							reward_hist = rand_batch[:,3]
							action_hist = rand_batch[:,0]


					if(total_steps == 0):
						sys.exit()

					curr_state = new_state

					total_steps+=1

def main():

	model = dqn(4,2)
	model.train()

if __name__ == "__main__":
	main()