import tensorflow as tf
import numpy as np
import sys
import random
import time
import imageio

from PIL import Image
from scipy.misc import imresize

sys.path.append('game/')



import wrapped_flappy_bird as game


def play():


	for i in range(1):

		game_state = game.GameState()

		# sys.exit()

		# frames = []
		# num_frames = 0

		for j in range(150):
			
			temp = random.randint(0,1)
			action = np.zeros([2])
			action[temp] = 1
			new_state, reward, done = game_state.frame_step(action)
			
			if(j%10 == 0):
				print(new_state)
			# frames.insert(num_frames, Image.fromarray(np.uint8(new_state)))
			# num_frames+=1

			if done:
				break

		sys.exit()


play()
