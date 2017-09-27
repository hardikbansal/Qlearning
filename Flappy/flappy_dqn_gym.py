import tensorflow as tf
import numpy as np
import sys
import random
import time
import imageio
import gym
import gym_ple


from PIL import Image
from scipy.misc import imresize

env = gym.make('FlappyBird-v0')



class network():

	def __init__(self, img_width, img_height, name="network"):

		self.name = name
		self.img_width = img_width
		self.img_height = img_height

	def net(self):

		with tf.variable_scope(self.name) as scope:

			self.input_state = tf.placeholder(tf.float32, [None, self.img_width, self.img_height, 4], name="input_state")

			o_c1 = general_conv2d(self.input_state, 32, 8, 8, 4, 4, padding="SAME", do_norm=False, name="conv1")
			o_c2 = general_conv2d(o_c1, 64, 4, 4, 2, 2, padding="SAME", do_norm=False, name="conv2")
			o_c3 = general_conv2d(o_c1, 64, 3, 3, 1, 1, padding="SAME", do_norm=False, name="conv3")

			shape = o_c3.get_shape().as_list()

			o_c3 = tf.reshape(o_c3,[-1, shape[1]*shape[2]*shape[3]])

			shape = o_c3.get_shape().as_list()

			o_l1 = linear1d(o_c3, shape[1], 512)

			self.output_weights = linear1d(o_l1, 512, 2)



class flappy():

	def __init__(self):

		# Defining the hyper parameters

		self.img_width = 80
		self.img_height = 80
		self.img_depth = 4
		self.img_size = self.img_width*self.img_height*self.img_depth
		self.eps = 0.1

		self.num_episodes = 10000
		self.pre_train_steps = 10000
		self.update_freq = 100
		self.batch_size = 32
		self.gamma = 0.9
		self.lr = 0.0001
		self.max_steps = 10000

	def copy_network(self, net1, net2):

		vars1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, net1.name)
		vars2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, net2.name)

		for idx in range(len(vars1)):
			vars2[idx].assign(vars1[idx])

	def model(self):

		self.main_net = network(self.img_width, self.img_height, name="main_net")
		self.target_net = network(self.img_width, self.img_height, name="target_net")

		# Initialising the Networks
		self.main_net.net()
		self.target_net.net()

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

	def pre_process(self, img):

		grey_matrix = np.array([0.2125, 0.7154, 0.0721])
		new_img = np.dot(img, grey_matrix)
		new_img = imresize(new_img, [80, 80])
		return new_img


	def train(self):


		self.model()

		#sys.exit()
		
		init = tf.global_variables_initializer()

		
		with tf.Session() as sess:

			sess.run(init)
			print("Initialized the model")
			# self.copy_network(self.main_net, self.target_net)
			
			total_steps = 0
			total_reward_list = []
			hist_buffer = []

			for i in range(self.num_episodes):

				# print("Started the episode " + str(i))

				curr_state = env.reset()
				img_batch = []
				total_reward = 0

				for j in range(4):
					temp_action = random.randint(0,1)
					new_state, reward, done, _ = env.step(temp_action)

					temp_img = self.pre_process(new_state)
					img_batch.insert(len(img_batch), temp_img)

					total_reward += reward
				
				# sys.exit()

				while(True):

					temp = random.random()
					
					if temp < self.eps :
						temp_action = random.randint(0,1)
					else :
						temp_weights = sess.run([self.main_net.output_weights], feed_dict={self.main_net.input_state:np.reshape(np.stack(img_batch,axis=2),[-1, 80, 80, 4])})
						temp_action = np.argmax(temp_weights)
					
					
					if(total_steps > self.pre_train_steps):
						self.eps = 0.99

					new_state, reward, done, _ = env.step(temp_action)
					temp_img = self.pre_process(new_state)

					total_reward += reward

					# print(type(hist_buffer))

					new_img_batch = img_batch[1:]
					new_img_batch.insert(3,temp_img)

					hist_buffer.append((np.stack(img_batch, axis=2), temp_action, reward, np.stack(new_img_batch,axis=2), done))
					
					if(len(hist_buffer) >= 50000):
						hist_buffer.pop(0)


					# Adding the image to the batch

					img_batch.insert(len(img_batch), temp_img)
					img_batch.pop(0)

					# Breaking the loop if the state is terminated

					if (done):
						break

					if(total_steps > self.pre_train_steps):

						rand_batch = random.sample(hist_buffer, self.batch_size)

						reward_hist = [m[2] for m in rand_batch]
						state_hist = [m[0] for m in rand_batch]
						action_hist = [m[1] for m in rand_batch]
						next_state_hist = [m[3] for m in rand_batch]

						# print(len(state_hist))

						# sys.exit()

						temp_target_q = sess.run(self.main_net.output_weights, feed_dict={self.main_net.input_state:np.stack(next_state_hist)})

						# sys.exit()

						temp_target_q = np.amax(temp_target_q,1)
						temp_target_reward = reward_hist + self.gamma*temp_target_q
						temp_target_reward =  np.reshape(temp_target_reward, [self.batch_size, 1])
						
						
						# print((action_hist))

						_ = sess.run(self.loss_opt, feed_dict={self.main_net.input_state:np.stack(state_hist), self.target_reward:temp_target_reward, self.action_list:np.reshape(np.stack(action_hist),[self.batch_size, 1])})
						
						# sys.exit()


					total_steps+=1
				
				print("Total rewards in episode " + str(i) + " is " + str(total_reward))
				sys.exit()
			# for var in self.model_vars: print(var.name, sess.run(var.name))

			self.play(sess)


	def play(self):

		for i in range(1):

			curr_state = env.reset()
			env.render()

			total_steps = 0

			for j in range(self.max_steps):
				
				temp = random.randint(0,1)
				action = np.zeros([2])
				action[temp] = 1
				new_state, reward, done, _ = env.step(temp)

				total_steps += 1
				
				if done:
					break

			print("Total Steps ", str(total_steps))

			sys.exit()



def main():

	model = flappy()
	# model.train()
	model.play()

if __name__ == "__main__":
	main()
