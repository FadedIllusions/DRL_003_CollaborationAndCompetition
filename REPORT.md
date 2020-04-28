### Continuous Control Report

This project implements a multi-agent Deep Deterministic Policy Gradient (DDPG) to solve the Unity ML Agents "[Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis)" environment. See [Continuous Control with Deep Reinforcement Learning](https://arxiv.org/pdf/1509.02971.pdf) for more information.

<p align="center">
    <img src="/images/ddpg_algorithm.png" alt="DDPG Algorithm">
</p>


# Deep Deterministic Policy Gradient

Actor: 
  Input(Dims:2) => FC(Dims:400) => FC(Dims:300) => Output(Dims:24) using [ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) as the activation function of the two hidden layers and Hyperbolic Tangent ([Tanh](https://en.wikipedia.org/wiki/Hyperbolic_functions#Tanh)) activation function on the output layer. Wherein the input dimensions correspond to the number of dimensions in the state space and the output dimensions correspond to the number of dimensions in the action space.

Critic:
  Input(Dims:2) => FC(Dims:256) => FC(Dims:256) => Output(Dims:24), also using ReLU but without an activation layer on the output, as it's to approximate a single scalar value 'maximizer' over the q-values of the following state using the Bellman equation.

As with our previous DQN (seen in [DRL_001_Navigation](https://github.com/FadedIllusions/DRL_001_Navigation)), our DDPG contains a replay memory/buffer used to record agent experience so as to be able to randomly sample for additional training, which tends to be especially useful in rarely occuring states. 

Unlike in the DQN, we are also implementing 'soft (target) updates', as seen in "Continuous Control with Deep Reinforcement Learning. In DQN, we have two copies of our update weights -- one of our regular network and one of our target network, copying our regular network weights into our target network weights every 10k timesteps. However, when implementing a 'soft update' in DDPG, we have two copies for each network -- a regular and a target for both the actor and the critic. As we progress through the timesteps, we slowly 'blend' our regular weights with our target weights for a quicker convergence. (In example, we could 'mix' 0.01% of our regular network weights with out target network weights every time step.)


# Hyperparameters

```python
BUFFER_SIZE = int(1e5)    # Replay Buffer Size
BATCH_SIZE = 128          # Minibatch Size
GAMMA = 0.99              # Discount Factor
TAU = 3e-1                # Soft Update Of Target Parameters
LR_ACTOR = 1e-4           # Actor Learning Rate
LR_CRITIC = 1e-4          # Critic Learning Rate
WEIGHT_DECAY = 0          # L2 Weight Decay

```

# Plot Of Rewards

![Rewards Plot](/images/scores_525.png)

![Network Feedback](/images/solved_525.png)


# Ideas For Future Work

I'd like to further tune the hyperparameters and look into different forms of learning rate optimization for DDPG networks, as well as attempt to implement PPO and A3C.
