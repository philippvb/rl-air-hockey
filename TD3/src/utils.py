

import numpy as np
import torch
import os, sys

class ReplayBuffer(object):
	"""Implementation of a replay buffer taken from https://github.com/sfujim/TD3, and modified at points annotated with ADDED
	"""
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.done = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.done[self.ptr] = done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.done[ind]).to(self.device)
		)

	def reset(self):
		"""ADDED
			Resets the replay buffer
		"""
		self.ptr = 0
		self.size = 0
		self.state = np.zeros((self.max_size, self.state_dim))
		self.action = np.zeros((self.max_size, self.action_dim))
		self.next_state = np.zeros((self.max_size, self.state_dim))
		self.reward = np.zeros((self.max_size, 1))
		self.done = np.zeros((self.max_size, 1))

	def save(self, path):
		if not os.path.exists(path):
			os.mkdir(path)
		
		torch.save(self.state, path + "/states")
		torch.save(self.action, path + "/action")
		torch.save(self.next_state, path + "/next_state")
		torch.save(self.reward, path + "/reward")
		torch.save(self.done, path + "/done")

	def load(self, path):
		self.state = torch.load(path + "/states")
		self.actions = torch.load(path + "/action")
		self.next_state = torch.load(path + "/next_state")
		self.reward = torch.load(path + "/reward")
		self.done = torch.load(path + "/done")


class HiddenPrints:
    """Class to hide the prints.
    """
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def moving_average(x, N):
	"""Helper method for computing a moving average

	Args:
		x (np.array): The data
		N (int): size of the window to use

	Returns:
		np.array: moving average of the data
	"""
	cumsum = np.cumsum(np.insert(x, 0, 0)) 
	return (cumsum[N:] - cumsum[:-N]) / float(N)
		

