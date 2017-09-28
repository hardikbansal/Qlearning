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

		if(temp > self.bandits[state][index]):
			return -1
		else :
			return 1

	def model_setup(self):

		self.input_state = tf.placeholder(shape=[self.num_bandits], dtype=tf.int32)

		self.weights = tf.Variable(tf.ones([self.num_bandits, self.num_hands]))

		output_weights = tf.matmul(tf.reshape(tf.cast(self.input_state, dtype=tf.float32),[1,-1]),self.weights)
		self.output_move = tf.argmax(output_weights,1)

	def loss_setup(self):

		self.curr_move = tf.placeholder(shape=[1],dtype=tf.int32)
		self.curr_reward = tf.placeholder(shape=[1],dtype=tf.float32)
		self.curr_state = tf.placeholder(shape=[1],dtype=tf.int32)
		
		loss = -tf.log(self.weights[self.curr_state[0], self.curr_move[0]])*self.curr_reward

		optimizer = tf.train.AdamOptimizer(0.001, beta1=0.5)
		self.loss_optimizer = optimizer.minimize(loss)


	def train(self):

		self.model_setup()
		self.loss_setup()

		init = tf.global_variables_initializer()

		with tf.Session() as sess:

			sess.run(init)

			for step in range(10000):

				state = np.random.randint(self.num_bandits, size=1)[0]
				state_oh = np.eye(self.num_bandits)[state]
				temp = np.random.uniform(0,1)
				if temp > self.e:
					action = sess.run(self.output_move, feed_dict={self.input_state:state_oh})
				else:
					# action = sess.run(self.output_move, feed_dict={self.input_state:state_oh})
					action = np.random.randint(self.num_hands,size=1)

				action = action.astype(int)
				# print(action.shape)
				reward = self.get_reward(state, action[0])
				_, temp_weight = sess.run([self.loss_optimizer, self.weights] , feed_dict={self.curr_state:[state], self.curr_move:action, self.curr_reward:[reward]})
				if(step % 100 == 0):
					print(temp_weight)


def main():

	model = contextual_bandits()
	model.train()

if __name__ == "__main__":
	main()
