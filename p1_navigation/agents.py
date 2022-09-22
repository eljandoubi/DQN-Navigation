# This code is based on the code of DQN coding exercice of udacity nanodegree deep reinforcement learning.

import numpy as np
from collections import namedtuple, deque


import torch
import torch.nn.functional as F


BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
UPDATE_EVERY = 4        # how often to update the network
EPS=1e-10

class AgentQ():
    """Interacts with and learns from the environment."""

    def __init__(self, qnetwork_local, qnetwork_target, optimizer, param_opt,
                 action_size=4, seed=0, device="cpu", dqn="vanilla",
                 buffer_size=BUFFER_SIZE,batch_size=BATCH_SIZE,a=0,b=0):
        
        """Initialize an Agent object.
        
        Params
        ======
            qnetwork_local (nn.Module): local network
            qnetwork_local (nn.Module): target network
            optimizer (torch.optim.optimizer) : the optimizer of networks
            param_opt (dict): dictionary of the optimizer parameters
            action_size (int): dimension of each action
            seed (int): random seed
            device (torch.device): cpu or cuda
            dqn ("vanilla","double"): DQN version
            buffer_size  (int): replay buffer size
            batch_size (int): minibatch size
            a (float): priority power
            b (float): adjustment power
        """
        
        assert dqn in ["vanilla","double"]
        
        self.dqn=dqn
        self.action_size = action_size
        self.seed = np.random.seed(seed)
        self.device=device
        self.batch_size=batch_size

        # Network
        self.qnetwork_local = qnetwork_local.to(device)
        self.qnetwork_target = qnetwork_target.to(device)
        self.optimizer = optimizer(self.qnetwork_local.parameters(),**param_opt)

        # Replay memory
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, seed, device, a, b)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        
            
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        
        # If enough samples are available in memory, get random subset and learn
        if len(self.memory) > self.batch_size:
        
            if self.t_step == 0:
                
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)
                
                # ------------------- update target network ------------------- #
                self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)
            

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if np.random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy()).astype(np.int32)
        else:
            return np.random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        
        states, actions, rewards, next_states, dones, adjs = experiences
        
        
        if self.dqn=="vanilla":
            # Get max predicted Q values (for next states) from target model
            Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        elif self.dqn=="double":
            # Get best action with respect to local model
            best_actions = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
            
            # Compute Q targets for next states 
            Q_targets_next = self.qnetwork_target(next_states).detach().gather(1, best_actions)
            
            
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        
        # Update transition priority
        self.memory.update_prioritty((Q_targets-Q_expected).detach().squeeze())
        

        # Compute loss
        loss = F.mse_loss(adjs*Q_expected, adjs*Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
                               

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, device, a=0, b=0):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            a (float): priority power
            b (float): adjustment power
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = np.random.seed(seed)
        self.device=device
        self.buffer_size=buffer_size
        self.a=a
        self.b=b
        self.max_priority=1.
        self.weight = deque(maxlen=buffer_size)
        self.sum_weight=torch.tensor([0.]).to(device)
        self.max_adj=torch.tensor([0.]).to(device)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        
        self.weight.append(self.max_priority**self.a)
        self.sum_weight+=self.weight[-1]
        
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        
        # calcul prioritised probability
        self.weight=torch.FloatTensor(self.weight).to(self.device)
        prob=self.weight/self.sum_weight 
        
        l=len(self)

        self.idx=np.random.choice(np.arange(l), size=self.batch_size, p=prob.cpu().numpy(),replace=False) # get the sample indexes and store them
        experiences = [self.memory[i] for i in self.idx]
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        
        #evaluate normalised adjustment
        adj=(l*prob[self.idx])**(-0.5*self.b)
        max_adjs=torch.cummax(torch.cat((self.max_adj,adj)),dim=0)[0][1:]
        adj/=max_adjs
        self.max_adj=max_adjs[-1:]
        
        return (states, actions, rewards, next_states, dones, torch.unsqueeze(adj,1))

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    
    def get_prob(self):
        """Return the probability."""
        weights=torch.FloatTensor(self.weight).to(self.device)
        return weights/self.sum_weight
    
    def update_prioritty(self,delta):
        """Update transition priority."""
    
        delta=delta.abs()+EPS
        
        self.max_priority=max(self.max_priority,delta.max().item())
        
        delta=delta**self.a
        
        self.sum_weight+=(delta-self.weight[self.idx]).sum()
        
        self.weight[self.idx]=delta
        
        self.weight = deque(self.weight.cpu().numpy(),maxlen=self.buffer_size)

            
        
        
        
        
    
        