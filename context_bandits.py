import tensorflow as tf
import numpy as np
import sys

class  contextual_bandits(object):

	"""docstring for  contextual_bandits"""

	def __init__(self, arg):
		
		self.state = 0
		self.bandits = []

		self.num_bandits = len(self.bandits)
		self.e = 0.1

	def get_reward(self, state, index):

		temp = np.random.uniform(-1, 1)

		if(temp > bandits[state,index]):
			return -1
		else :
			return 1

	def model_setup():

		input_state = tf.placeholder([self.num_bandits], tf.int32)

		weights = tf.Variable(tf.ones([self.num_bandits, self.num_hands]))

		output_weights = tf.matmul(input_state,weights)
		output_move = tf.max(output_weights)

	def loss_setup():

		


	def train():

		init = tf.global_variables_initializer()

		with tf.Session() as sess:

			sess.run(init)


			for step in range(num_episodes):

				state = np.random.randint(self.num_bandits, size=1)[0]

				state_oh = np.eyes(self.num_bandits)[state]

				temp = np.random.uniform(0,1)

				if temp > e:
					action = sess.run([output_move], feed_dict={input_state = state})
				else:
					action = np.random.randint(self.num_hands,size=1)

