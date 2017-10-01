import tensorflow as tf
import numpy as np
import sys
import random
import time
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../utils"))

from layers import *

import gym
from gym import wrappers



class network():

	def __init__(self, state_size, action_size, h1_actor_size=400, h2_actor_size=300, h1_critic_size=400, h2_critic_size=300, name="actor_network"):

		self.state_size = state_size
		self.action_size = action_size
		self.name = name

		# Actor latent variables
		
		self.h1_actor_size = h1_actor_size
		self.h2_actor_size = h2_actor_size

		# Critic latent Variables

		self.h2_critic_size = h2_critic_size
		self.h1_critic_size = h1_critic_size


	def net(self):

		self.input_state = tf.placeholder(tf.float32, [None, self.state_size], name="input_state")
		
		with tf.variable_scope(self.name + "_actor") as scope:

			h1_actor = tf.nn.relu(linear1d(self.input_state, self.state_size, self.h1_actor_size, name="hidden1"))
			h2_actor = tf.nn.relu(linear1d(h1_actor, self.h1_actor_size, self.h2_actor_size, name="hidden2"))

			temp_out_action = tf.nn.tanh(linear1d(h2_actor, self.h2_actor_size, self.action_size, name="final"))
			self.out_action = temp_out_action*2

		with tf.variable_scope(self.name + "_critic") as scope:

			self.action = tf.placeholder(tf.float32, [None, self.action_size], name="action")

			h1_critic = tf.nn.relu(linear1d(self.input_state, self.state_size, self.h1_critic_size, name="hidden1"))

			weight_1 = tf.Variable(tf.truncated_normal([self.h1_critic_size, self.h2_critic_size], stddev=0.01), name="weight_1")
			weight_2 = tf.Variable(tf.truncated_normal([self.action_size, self.h2_critic_size], stddev=0.01), name="weight_2")
			bias = tf.Variable(tf.zeros([self.h2_critic_size]), name="bias")

			h2_critic = tf.matmul(h1_critic, weight_1) + tf.matmul(self.action, weight_2) + bias

			self.q_value = linear1d(h2_critic, self.h2_critic_size, 1, name="final")

class dqn():

	def __init__(self, state_size, action_size):

		# Defining the hyper parameters

		self.state_size = state_size
		self.action_size = action_size
		self.eps = 0.5

		self.num_episodes = 3000
		self.max_steps = 1000
		self.pre_train_steps = 10
		self.update_freq = 100
		self.batch_size = 10
		self.gamma = 0.9
		self.lr = 0.0001

	def copy_network(self, net1, net2):

		vars1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, net1.name)
		vars2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, net2.name)

		for idx in range(len(vars1)):
			vars2[idx].assign(vars1[idx])

	def model(self):

		self.main_net = network(self.state_size, self.action_size, name="main_net")

		# Initialising the Networks
		self.main_net.net()

		# Defining the model for the training

		self.target_reward = tf.placeholder(tf.float32, [None, 1], name="target_reward")
		self.action_list = tf.placeholder(tf.int32, [None, 1], name="action_list")

		# observed_reward = tf.reduce_sum(self.main_net.output_weights*tf.one_hot(tf.reshape(self.action_list,[-1]),2,dtype=tf.float32),1,keep_dims=True)

		# self.loss = tf.reduce_mean(tf.square(observed_reward - self.target_reward))

		# optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
		# self.loss_opt = optimizer.minimize(self.loss)

		# self.model_vars = tf.trainable_variables()
		# for var in self.model_vars: print(var.name)


	def policy(self, env, sess, state, algo="e_greedy"):
		if(algo == "e_greedy"):

			temp = random.random()

			if temp < self.eps :
				temp_action = [env.action_space.sample()]
			else :
				temp_action = sess.run([self.main_net.out_action], feed_dict={self.main_net.input_state:np.reshape(curr_state,[-1, self.state_size])})
				# print(temp_weights)

		return temp_action


	def train(self):


		env = gym.make('Pendulum-v0')

		self.model()

		init = tf.global_variables_initializer()

		with tf.Session() as sess:

			sess.run(init)
			print("Initialized the model")
			
			total_steps = 0
			total_reward_list = []
			hist_buffer = []

			for i in range(self.num_episodes):

				# print("Started the episode " + str(i))

				curr_state = env.reset()
				total_reward = 0

				for j in range(1, self.max_steps+1):

					temp = random.random()

					print(temp)

					if temp < self.eps :
						temp_action = [env.action_space.sample()]
					else :
						temp_action = sess.run([self.main_net.out_action], feed_dict={self.main_net.input_state:np.reshape(curr_state,[-1, self.state_size])})
						# print(temp_action)
						# sys.exit()

					self.eps*=0.99
					new_state, reward, done, _ = env.step(temp_action[0])
					total_reward += reward

					# print(reward)
					# print(type(hist_buffer))

					hist_buffer.append((curr_state, temp_action[0], reward, new_state, done))
					
					if(len(hist_buffer) >= 10000):
						hist_buffer.pop(0)

					curr_state = new_state

					if (done):
						break

					# sys.exit()
					if(total_steps > self.batch_size):

						# print("Training the network")

						rand_batch = random.sample(hist_buffer, self.batch_size)

						reward_hist = [m[2] for m in rand_batch]
						state_hist = [m[0] for m in rand_batch]
						action_hist = [m[1] for m in rand_batch]
						next_state_hist = [m[3] for m in rand_batch]

						temp_action = sess.run(self.main_net.out_action, feed_dict={self.main_net.input_state:np.vstack(next_state_hist)})
						temp_target_q = sess.run(self.main_net.q_value, feed_dict={self.main_net.input_state:np.vstack(next_state_hist), self.main_net.action:temp_action})
						# sys.exit()
						temp_target_reward = reward_hist + self.gamma*temp_target_q
						temp_target_reward =  np.reshape(temp_target_reward, [self.batch_size, 1])
						
						# print(action_hist.shape)

						_ = sess.run(self.loss_opt, feed_dict={self.main_net.input_state:np.vstack(state_hist), self.target_reward:temp_target_reward, self.action_list:np.vstack(action_hist)})


					total_steps+=1

					
				if(len(total_reward_list) == 10):
					# print("Here")
					total_reward_list.pop(0)
				total_reward_list.insert(len(total_reward_list), total_reward)

				avg_reward = sum(total_reward_list)/len(total_reward_list)

				if(avg_reward > 198):
					break
				
				print(avg_reward)

				
				print("Total rewards in episode " + str(i) + " is " + str(total_reward))
			
			# for var in self.model_vars: print(var.name, sess.run(var.name))

			self.play(sess, method="trained")


	def play(self, sess, method="random"):

		env = gym.make('Pendulum-v0')

		for i in range(10):
			
			curr_state = env.reset()
			total_reward = 0

			for j in range(self.max_steps):

				env.render()

				if(method == "random"):
					action = [env.action_space.sample()]
				else :
					action = sess.run(self.main_net.out_action, feed_dict={self.main_net.input_state:np.reshape(curr_state,[-1, self.state_size])})
				
				new_state, reward, done, _ = env.step(action[0])
				
				if(done == True):
					break
				
				total_reward += reward
				curr_state = new_state
			
			print("Total rewards in testing step " + str(i) + " are " + str(total_reward))



def main():

	model = dqn(3,1)
	model.train()
	# model.play(None)

if __name__ == "__main__":
	main()