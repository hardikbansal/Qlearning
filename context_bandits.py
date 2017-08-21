import tensorflow as tf
import numpy as np
import sys

class  contextual_bandits(object):

	"""docstring for  contextual_bandits"""

	def __init__(self, arg):
		
		self.state = 0
		self.bandits = []

	def get_reward(self, state, index):

		
