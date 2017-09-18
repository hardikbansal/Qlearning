import tensorflow as tf
import numpy as np
import sys
import random
import time
import imageio

from PIL import Image

from images2gif import writeGif

sys.path.append('game/')
sys.path.append('../')

from layers import *



import wrapped_flappy_bird as game

class network():

	def __init__(self, state_size, action_size, h1_size=128, h2_size=128, name="network"):

		self.state_size = state_size
		self.action_size = action_size
		self.name = name
		self.h1_size = h1_size
		self.h2_size = h2_size

	def net(self):

		with tf.variable_scope(self.name) as scope:

			self.input_state = tf.placeholder(tf.float32, [None, self.state_size], name="input_state")

			h1 = tf.nn.tanh(linear1d(self.input_state, self.state_size, self.h1_size, name="hidden1"))
			h2 = tf.nn.tanh(linear1d(h1, self.h1_size, self.h2_size, name="hidden2"))

			self.output_weights = linear1d(h2, self.h2_size, self.action_size, name="final")

			self.out_action = tf.argmax(self.output_weights, 1)

class flappy():

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
		# self.target_net = network(self.state_size, self.action_size, name="target_net")

		# Initialising the Networks
		self.main_net.net()
		# self.target_net.net()

		# Defining the model for the training

		self.target_reward = tf.placeholder(tf.float32, [None, 1], name="target_reward")
		self.action_list = tf.placeholder(tf.int32, [None, 1], name="action_list")

		observed_reward = tf.reduce_sum(self.main_net.output_weights*tf.one_hot(tf.reshape(self.action_list,[-1]),2,dtype=tf.float32),1,keep_dims=True)

		# print(self.target_reward.shape)
		# sys.exit()

		self.loss = tf.reduce_mean(tf.square(observed_reward - self.target_reward))

		optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
		self.loss_opt = optimizer.minimize(self.loss)

		self.model_vars = tf.trainable_variables()
		for var in self.model_vars: print(var.name)


	def train(self):

		env = game.GameState()

		self.model()

		init = tf.global_variables_initializer()

		with tf.Session() as sess:

			sess.run(init)
			print("Initialized the model")
			# self.copy_network(self.main_net, self.target_net)
			
			total_steps = 0
			total_reward_list = []
			hist_buffer = []

			for i in range(3000):

				# print("Started the episode " + str(i))

				curr_state = env.reset()
				total_reward = 0

				for j in range(1, self.max_steps+1):

					temp = random.random()
					
					if temp < self.eps :
						temp_action = [env.action_space.sample()]
					else :
						# print("I am here")
						temp_action, temp_weights = sess.run([self.main_net.out_action, self.main_net.output_weights], feed_dict={self.main_net.input_state:np.reshape(curr_state,[-1, self.state_size])})
						# print(temp_weights)
					
					self.eps*=0.99

					new_state, reward, done, _ = env.step(temp_action[0])

					total_reward += reward
					if(done):
						reward = -100

					# print(type(hist_buffer))

					hist_buffer.append((curr_state, temp_action[0], reward, new_state, done))
					if(len(hist_buffer) >= 10000):
						hist_buffer.pop(0)

					curr_state = new_state

					if (done):
						break

					if(total_steps > self.batch_size):

						# print("Training the network")

						rand_batch = random.sample(hist_buffer, self.batch_size)

						reward_hist = [m[2] for m in rand_batch]
						state_hist = [m[0] for m in rand_batch]
						action_hist = [m[1] for m in rand_batch]
						next_state_hist = [m[3] for m in rand_batch]

						temp_target_q = sess.run(self.main_net.output_weights, feed_dict={self.main_net.input_state:np.vstack(next_state_hist)})

						# sys.exit()

						temp_target_q = np.amax(temp_target_q,1)
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

			self.play(sess)


	def play(self):


		for i in range(self.num_episodes):

			game_state = game.GameState()

			# sys.exit()

			# frames = []
			# num_frames = 0

			for j in range(self.max_steps):
				
				temp = random.randint(0,1)
				action = np.zeros([2])
				action[temp] = 1
				new_state, reward, done = game_state.frame_step(action)
				
				# print(new_state)
				# frames.insert(num_frames, Image.fromarray(np.uint8(new_state)))
				# num_frames+=1

				if done:
					break

			sys.exit()












def main():

	model = flappy(4,2)
	# model.train()
	model.play()

if __name__ == "__main__":
	main()