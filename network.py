"""
	This file contains a neural network module for us to
	define our actor and critic networks in PPO.
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import glob
import sys
import random
import time
from typing import Text
import numpy as np
import math
import network as NETWORK
from numpy.core.fromnumeric import choose
import ENV as env
import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import MultivariateNormal
import matplotlib.pyplot as plt

class FeedForwardNN(nn.Module):
	"""
		A standard in_dim-64-64-out_dim Feed Forward Neural Network.
	"""
	def __init__(self, in_dim, out_dim):
		"""
			Initialize the network and set up the layers.   
 
			Parameters:
				in_dim - input dimensions as an int
				out_dim - output dimensions as an int

			Return:
				None
		"""
		super(FeedForwardNN, self).__init__()

		self.layer1 = nn.Linear(in_dim, 64)
		self.layer2 = nn.Linear(64, 64)
		self.layer3 = nn.Linear(64, out_dim)

	def forward(self, obs):
		"""
			Runs a forward pass on the neural network.

			Parameters:
				obs - observation to pass as input

			Return:
				output - the output of our forward pass
		"""
		# Convert observation to tensor if it's a numpy array
		if isinstance(obs, np.ndarray):
			obs = torch.tensor(obs, dtype=torch.float)
		# print("in network output is", obs.shape)
		activation1 = F.relu(self.layer1(obs))
		activation2 = F.relu(self.layer2(activation1))
		output = self.layer3(activation2)
		# print("in network output is", output)
		return output
# class ActorNetwork(nn.Module):
#     def __init__(self, n_actions, input_dims, alpha,
#                  fc1_dims=64, fc2_dims=64, chkpt_dir="CHKPT"):
#         super(ActorNetwork, self).__init__()

#         self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')
# #         self.checkpoint_file = chkpt_dir
#         self.actor = nn.Sequential(
#             nn.Linear(input_dims, fc1_dims),
#             nn.ReLU(),
#             nn.Linear(fc1_dims, fc2_dims),
#             nn.ReLU(),
#             nn.Linear(fc2_dims, n_actions)

#         )

#         # self.optimizer = optim.Adam(self.parameters(), lr=alpha)
#         # self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
#         # self.to(self.device)

#     def forward(self, state):
#         #         dist = self.actor(state)
#         #         dist = Categorical(dist)
#         mean = self.actor(state)
# #         dist = MultivariateNormal(mean, self.cov_mat)
#         print("mean shape",mean.shape)
#         return mean

#     def save_checkpoint(self):
#         T.save(self.state_dict(), self.checkpoint_file)

#     def load_checkpoint(self):
#         self.load_state_dict(T.load(self.checkpoint_file))


# class CriticNetwork(nn.Module):
#     def __init__(self, input_dims, alpha, fc1_dims=64, fc2_dims=64,
#                  chkpt_dir="CHKPT"):
#         super(CriticNetwork, self).__init__()
#         # print(input_dims)
#         self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')

#         self.critic = nn.Sequential(
#             nn.Linear(input_dims, fc1_dims),
#             nn.ReLU(),
#             nn.Linear(fc1_dims, fc2_dims),
#             nn.ReLU(),
#             nn.Linear(fc2_dims, 1)
#         )

#         self.optimizer = optim.Adam(self.parameters(), lr=alpha)
#         self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
#         # self.to(self.device)

#     def forward(self, state):
#         value = self.critic(state)

#         return value

#     def save_checkpoint(self):
#         T.save(self.state_dict(), self.checkpoint_file)

#     def load_checkpoint(self):
#         self.load_state_dict(T.load(self.checkpoint_file))
