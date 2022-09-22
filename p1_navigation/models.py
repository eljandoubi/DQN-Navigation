# This code is based on the code of DQN coding exercice of udacity nanodegree deep reinforcement learning.

import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
class ConvQN(nn.Module):
    """Actor (Policy) DQN Model."""
    def __init__(self,in_channel,action_size,seed):
        """Initialize parameters and build model.
        Params
        ======
            in_channel (int): Number of channel per state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(ConvQN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.seq=nn.Sequential(
            nn.Conv2d(in_channel, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512,action_size)
            )
        
    def forward(self,state):
        """Build a network that maps state -> action values."""
        return self.seq(state)
    
class DConvQN(nn.Module):
    """Actor (Policy) dueling DQN Model."""
    def __init__(self,in_channel,action_size,seed):
        """Initialize parameters and build model.
        Params
        ======
            in_channel (int): Number of channel per state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(DConvQN, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        
        self.seq1=nn.Sequential(
            nn.Conv2d(in_channel, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten()
            )
     
        
        self.seq2=nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,action_size)
            )
        
        self.seq3=nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,1)
            )
        
    def forward(self,state):
        """Build a network that maps state -> action values."""
        x=self.seq1(state)
        y=self.seq2(x)
        y=y-y.max(1,keepdim=True)[0]
        return self.seq3(x)+y