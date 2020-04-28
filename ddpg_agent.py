# Import Needed Packages
import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim



# Parameters
BUFFER_SIZE = int(1e5)    # Replay Buffer Size
BATCH_SIZE = 128          # Minibatch Size
GAMMA = 0.99              # Discount Factor
TAU = 3e-1                # Soft Update Of Target Parameters
LR_ACTOR = 1e-4           # Actor Learning Rate
LR_CRITIC = 1e-4          # Critic Learning Rate
WEIGHT_DECAY = 0          # L2 Weight Decay


# Set Device To GPU Unless Unavailable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class Agent():
    """
    Interacts With And Learns From Environment.
    """
    
    def __init__(self, state_size, action_size, num_agents, random_seed):
        """
        Init Agent Object
        
        Parameters:
        -----------
            state_size (int): Dimension Of Each State
            action_size (int): Dimension Of Each Action
            num_agents (int): Number Of Agents
            random_seed (int): Random Seed
        """
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.num_agents = num_agents
        
        # Actor Network (W/ Target Network)
        self.actor_local = Actor(state_size,action_size,random_seed).to(device)
        self.actor_target = Actor(state_size,action_size,random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(),lr=LR_ACTOR)
        
        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size,action_size,random_seed).to(device)
        self.critic_target = Critic(state_size,action_size,random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(),lr=LR_CRITIC,weight_decay=WEIGHT_DECAY)
        
        # Noise Process
        self.noise = OUNoise(action_size, random_seed)
        
        # Replay Memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        
        
    def step(self, state, action, reward, next_state, done, timestep):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        for i in range(self.num_agents):
            self.memory.add(state[i,:], action[i,:], reward[i], next_state[i,:], done[i])

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE and timestep % 25 == 0:
            for _ in range(15):
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)
                
                
    def act(self, states, add_noise=False):
        """
        Returns Actions For Given State As Per Current Policy.
        """
        
        states = torch.from_numpy(states).float().to(device)
        self.actor_local.eval()
        actions = []

        with torch.no_grad():
            for state in states:
                # Action For Each Agent
                action = self.actor_local(state).cpu().data.numpy()
                actions.append(action)
            
        self.actor_local.train()
        
        if add_noise:
            actions += self.noise.sample()
            
        return np.clip(actions, -1, 1)
    
    
    def reset(self):
        self.noise.reset()
        
        
    def learn(self, experiences, gamma):
        """
        Update Policy And Value Parameters Using Given Batch Of Experience Tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Parameters:
        -----------
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        
        states, actions, rewards, next_states, dones = experiences
        
        
        # ------------------------- UPDATE CRITIC ------------------------- #
        # Get Predicted Next-State Actions And Q-Values From Target Models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        
        # Compute Q Targets For Current States (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        
        # Compute Critic Loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize Loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        
        # ------------------------- UPDATE ACTOR ------------------------- #
        # Compute Actor Loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        
        # Minimize The Loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        
        # -------------------- UPDATE TARGET NETWORKS -------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)
        
        
    def soft_update(self, local_model, target_model, tau):
        """
        Soft Update Model Parameters.
        
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Parameters:
        -----------
            local_model: PyTorch Model (Weights Will Be Copied From)
            target_model: PyTorch Model (Weights Will Be Copied To)
            tau (float): Interpolation Parameter 
        """
        
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
            
            
class OUNoise:
    """
    Ornstein-Uhlenbeck Process.
    """
    
    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """
        Init Parameters And Noise Process.
        """
        
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()
        
        
    def reset(self):
        """
        Reset Internal State (= Noise) To Mean (mu).
        """
        
        self.state = copy.copy(self.mu)
        
    
    def sample(self):
        """
        Update Internal State And Return As Noise Sample.
        """
        
        x = self.state
        dx = self.theta * (self.mu -x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        
        return self.state
    
    
    
class ReplayBuffer:
    """
    Fixed-Size Buffer To Store Experience Tuples.
    """
    
    def __init__(self, action_size, buffer_size, batch_size, seed):
        """
        Init ReplayBuffer Object.
        
        Parameters:
        -----------
        buffer_size (int): Max Size Of Buffer
        batch_size (int): Size Of Each Training Batch
        """
        
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    
    def add(self, state, action, reward, next_state, done):
        """
        Add New Experience To Memory.
        """
        
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    
    def sample(self):
        """
        Randomly Sample Batch Of Experiences From Memory.
        """
        
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    
    def __len__(self):
        """
        Return the current size of internal memory.
        """
        
        return len(self.memory)