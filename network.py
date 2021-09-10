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
    def __init__(self, in_dim_1, in_dim_2, out_dim, type):
        super(FeedForwardNN, self).__init__()
        self.type = type

        # Common preprocess layers for both actor and critic
        # Mostly we will process the larger data like: image/radar here
        self.radar_layer1 = nn.Linear(in_dim_1,512)
        self.radar_layer2 = nn.Linear(512, 256)
        self.radar_layer3 = nn.Linear(256, 128)
        self.radar_layer4 = nn.Linear(128, 64)

        self.state_layer1 = nn.Linear(in_dim_2,4)
        self.state_layer2 = nn.Linear(4, 16)

        if type == "actor":
            # Actor Layers
            self.actor1 = nn.Linear(80, 64)   # 128+64
            self.actor2 = nn.Linear(64, 32)
            self.actor3 = nn.Linear(32, out_dim)

        elif type == "critic":
            # Critic Layers
            # self.action1 = nn.Linear(1, 4)
            # self.action2 = nn.Linear(4, 16)

            self.critic1 = nn.Linear(80, 64)  # 128+64+32
            self.critic2 = nn.Linear(64, 32)
            self.critic3 = nn.Linear(32, out_dim)

        else:
            print("ERROR-----ERROR-----ERROR")



    def forward(self, obs_radar, obs_state, obs_action):
        # In case we pass in direct np.array
        # then we need to convert to torch
        if isinstance(obs_radar,np.ndarray):
            obs_radar = torch.tensor(obs_radar,dtype=torch.float)

        if isinstance(obs_state,np.ndarray):
            obs_state = torch.tensor(obs_state,dtype=torch.float)


        # Basic preprocess for both actor & critic 
        # RADAR
        act_radar1 = F.relu(self.radar_layer1(obs_radar))
        act_radar2 = F.relu(self.radar_layer2(act_radar1))
        act_radar3 = F.relu(self.radar_layer3(act_radar2))
        radar = F.relu(self.radar_layer4(act_radar3))

        # STATE
        act_state1 = F.relu(self.state_layer1(obs_state))
        state = F.relu(self.state_layer2(act_state1))

        preprocess = torch.cat((radar, state), -1)


        # Actor Module
        if self.type == "actor":
            act1 = F.relu(self.actor1(preprocess))
            act2 = F.relu(self.actor2(act1))
            output = torch.tanh(self.actor3(act2))

        # Critic Module
        elif self.type == "critic":
            # act1 = F.relu(self.action1(obs_action))
            # action = F.relu(self.action2(act1))

            # concat = torch.cat((preprocess, action), 1)

            out1 = F.relu(self.critic1(preprocess))
            out2 = F.relu(self.critic2(out1))
            output = self.critic3(out2)

        else:
            print("ERROR-----ERROR-----ERROR")

        return output