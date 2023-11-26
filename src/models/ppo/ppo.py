import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from typing import Callable
from torch.distributions.normal import Normal
from gymnasium import spaces


def make_env(env_: gym.Env, seed:int) -> Callable[[], gym.Env]:
    def thunk():
        #env = gym.make(gym_id)
        env = gym.wrappers.RecordEpisodeStatistics(env_)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Weight initializer for the layer"""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs, input_dims, output_dims):
        """
        Initializes the agent with actor and value network.
        We only implement the continuous actor network. 
        The parameterizing tensor is a mean and standard deviation vector, 
        which parameterize a gaussian distribution.
        
        Args:
        - input_dims: the input dimension of the value and actor network (i.e dimension of state = 8+8+1=17)
        - output_dims: output dimension of the actor network (i.e. dimension of actions = 8)
        
        """
        super(Agent, self).__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(input_dims, 64)), #np.array(envs.single_observation_space.shape).prod()
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(input_dims, 64)), #np.array(envs.single_observation_space.shape).prod()
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, output_dims), std=0.01), #np.prod(envs.single_action_space.shape)=8
        )
        #state independent standard deviations
        self.actor_logstd = nn.Parameter(torch.zeros(1, output_dims)) #np.prod(envs.single_action_space.shape)
        self.envs = envs

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        """
        Performs inference using the value network.
        Inputs:
        - x, the state passed in from the agent
        Returns:
        - The scalar (float) value of that state, as estimated by the net
        """
        #assert isinstance(self.envs.single_action_space, spaces.Box) == True     
        #print("self.actor_mean(x)", self.actor_mean(x).shape) [1,8]
        action_mean = self.actor_mean(x)
        #action_mean = torch.reshape(self.actor_mean(x), (1,-1))
        #print("action_mean shape", action_mean.shape) [1,8]
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        #print("action space", self.envs.action_space)
        #print("single action space", self.envs.single_action_space)
        if action is None:
            action = probs.sample()
            # Actions could be on arbitrary scale, so clip the actions to avoid
            # out of bound error (e.g. if sampling from a Gaussian distribution)
        
        #clip action, see https://ai.stackexchange.com/questions/28572/how-to-define-a-continuous-action-distribution-with-a-specific-range-for-reinfor
        action = np.clip(action, self.envs.single_action_space.low, self.envs.single_action_space.high)
        assert torch.all((action >= -1) & (action <= 1)), "Not all elements are in the interval [-1, 1]"

        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


