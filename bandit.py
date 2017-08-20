import tensorflow as tf
import numpy as np
import sys


# Defining variables

e = 0.5

# Here we will be learning the Policy gradient method

# Loss = log(pi)*A
# where pi is the weight of that action
# and A is the reward for that action above the baseline


bandits = [0.2,0.8,-0.2,0.6]

# We would like to generate the reward that will be more for 
# more positive number. So, out function can be a random function
# that will return 1 with the prob corresponding to the number it is.

# So, what we can do it take a random number in between -1,1 and return 1 if the number 
# is greater than corresponding bandit and -1 otherwise.

def reward_bandit(index):

	temp = np.random.uniform(-1, 1)

	if(temp > bandits[index]):
		return -1
	else:
		return 1

# So, now that we have defined the reward function. We can define the 
# tensorflow graph.


weights = tf.Variable(tf.ones([len(bandits)]), name="weights")
sugg_move = tf.argmax(weights,0)
curr_move = tf.placeholder(shape=[1],dtype=tf.int32)
curr_reward = tf.placeholder(shape=[1],dtype=tf.float32)

print(curr_move.shape)


loss = -tf.log(weights[curr_move[0]])*curr_reward
# sys.exit()

optimizer = tf.train.AdamOptimizer(0.001, beta1=0.5)
loss_optimizer = optimizer.minimize(loss)



init = tf.global_variables_initializer()

with tf.Session() as sess:

	sess.run(init)

	for step in range(1000):

		temp = np.random.random(1)

		if temp > e :
			action = sess.run([sugg_move], feed_dict={})
		else :
			action = np.random.randint(len(bandits), size=1)

		reward = reward_bandit(action[0])

		_, temp_weight = sess.run([loss_optimizer, weights], feed_dict={curr_move:action , curr_reward:[reward]})

		# if(step%10 == 0) :

		print(temp_weight)
			



