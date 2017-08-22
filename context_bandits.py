import tensorflow as tf
import numpy as np
import sys

class  contextual_bandits():

	"""docstring for  contextual_bandits"""

	def __init__(self):
		
		self.state = 0
		self.bandits = [[0.2, 0.1, -0.3, 0.5], [0.1, 0.9, 0.1, -0.9], [-0.3, -0.1, 0.0, 1.0]]

		self.num_bandits = len(self.bandits)
		self.num_hands=4
		self.e = 0.1

	def get_reward(self, state, index):

		temp = np.random.uniform(-1, 1)

		if(temp > bandits[state,index]):
			return -1
		else :
			return 1

	def model_setup(self):

		input_state = tf.placeholder(shape=[self.num_bandits], dtype=tf.int32)

		self.weights = tf.Variable(tf.ones([self.num_bandits, self.num_hands]))

		output_weights = tf.matmul(tf.reshape(tf.cast(input_state, dtype=tf.float32),[1,-1]),self.weights)
		output_move = tf.argmax(output_weights,1)

	def loss_setup(self):

		curr_move = tf.placeholder(shape=[1],dtype=tf.int32)
		curr_reward = tf.placeholder(shape=[1],dtype=tf.float32)
		curr_state = tf.placeholder(shape=[1],dtype=tf.int32)
		
		loss = -tf.log(self.weights[curr_state, curr_move])*curr_reward

		optimizer = tf.train.AdamOptimizer(0.001, beta1=0.5)
		self.loss_optimizer = optimizer.minimize(loss)


	def train(self):

		self.model_setup()
		self.loss_setup()

		init = tf.global_variables_initializer()

		with tf.Session() as sess:

			sess.run(init)

			for step in range(num_episodes):

				state = np.random.randint(self.num_bandits, size=1)[0]
				state_oh = np.eyes(self.num_bandits)[state]
				temp = np.random.uniform(0,1)
				if temp > e:
					action = sess.run([output_move], feed_dict={input_state:state})
				else:
					action = np.random.randint(self.num_hands,size=1)

				reward = self.get_reward(state, action[0])
				_, temp_weight = sess.run([self.loss_optimizer, self.weights] , feed_dict={curr_state:[state], curr_move:action, curr_reward:reward})

				print(temp_weight)


def main():

	model = contextual_bandits()
	model.train()

if __name__ == "__main__":
	main()
