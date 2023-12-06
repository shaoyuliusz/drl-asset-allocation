import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from collections import namedtuple
from typing import NamedTuple, Callable
from torch.distributions.normal import Normal

def dict_to_namedtuple(dictionary:dict, name:str='Params') -> NamedTuple:
    """convert nested dictionaries to named tuples"""
    for key, value in dictionary.items():
        if isinstance(value, dict):
            # Recursively convert nested dictionaries to named tuples
            dictionary[key] = dict_to_namedtuple(value, f'{name}_{key.capitalize()}')
    return namedtuple(name, dictionary.keys())(**dictionary)


def replace_dict(dict1: dict[dict[str]], dict2: dict[str]) -> None:
    # Replace values in (nested) dict1 with values from dict2, return a new dict object
    dict_ = dict1.copy()
    for key in dict_:
        for k in dict_[key]:
            if k in dict2:
                dict_[key][k] = dict2[k]
    return dict_


def DRL_prediction(df_test: pd.DataFrame, model, env: gym.Env, test_obs) -> np.array:
    """
    run inference using trained deep reinforcement learning algorithm.
    Args:
        model: trained DRL model
        data: 
        env: test environment
        obs: test observations, np.array shape = (1, n_stocks)

    Returns:
        an action np.array representing the output portfolio vector ω_t every day, 
        of shape (len(data) X n_stocks)
    """
    if type(test_obs) == np.ndarray or type(test_obs) == list: #convert test_obs to tensor
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        test_obs = torch.Tensor(test_obs).to(device)

    for i in range(len(df_test.index.unique())):
        #get action from the trained model
        action, _, _, _ = model.get_action_and_value(test_obs)
        obs, rewards, dones, _, info = env.step(action.cpu().numpy())
           
        if i == (len(df_test.index.unique()) - 2): #351
            #access the original environment
            _env = env.env_fns[0]()
            actions_memory = _env.save_action_memory()
    
    return actions_memory #this must be (days X n_stocks)


def make_env_test(env_: gym.Env, seed:int, use_normalization: bool = True):
    def thunk():
        #env = gym.make(gym_id)
        #env = gym.wrappers.RecordEpisodeStatistics(env_)
        if use_normalization:
            env = gym.wrappers.ClipAction(env_)
            env = gym.wrappers.NormalizeObservation(env) #observation scaling
            env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, 0, 10)) #observation clipping
            env = gym.wrappers.NormalizeReward(env) #reward scaling
            env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10)) #reward clipping
            env.set_seed(seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env
        else:
            env_.set_seed(seed)
            env_.action_space.seed(seed)
            env_.observation_space.seed(seed)
            return env_
    return thunk

def make_env(env_: gym.Env, seed:int, use_normalization: bool=True) -> Callable[[], gym.Env]:
    """
    Args:
        env_: an instance of type gym.Env, StockEnvTrade
        seed: seed for the environment
        use_normalization: if normalize the environment.

    Returns:
        thunk: a function that returns the environment with gym wrappers
    """
    def thunk():
        if use_normalization:
            #env = gym.make(gym_id)
            env = gym.wrappers.RecordEpisodeStatistics(env_)
            env = gym.wrappers.ClipAction(env)
            env = gym.wrappers.NormalizeObservation(env) #observation scaling
            env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, 0, 10)) #observation clipping
            env = gym.wrappers.NormalizeReward(env) #reward scaling
            env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10)) #reward clipping
            env.set_seed(seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env
        else:
            env_.set_seed(seed)
            env_.action_space.seed(seed)
            env_.observation_space.seed(seed)
            return env_
    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Weight initializer for the layer"""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, input_dims, output_dims):
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
            layer_init(nn.Linear(input_dims, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(input_dims, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, output_dims), std=0.01),
        )
        #state independent standard deviations
        self.actor_logstd = nn.Parameter(torch.zeros(1, output_dims))


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

        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std) #Creates a normal distribution parameterized by action_mean and action_std

        if action is None:
            action = probs.sample()

        # Actions could be on arbitrary scale, so clip the actions to avoid
        # out of bound error (e.g. if sampling from a Gaussian distribution)
        # this is dealt with the environment wrapper ClipAction
        #clip action, see https://ai.stackexchange.com/questions/28572/how-to-define-a-continuous-action-distribution-with-a-specific-range-for-reinfor
        #action = np.clip(action, self.envs.single_action_space.low, self.envs.single_action_space.high)
        #assert torch.all((action >= -1) & (action <= 1)), "Not all elements are in the interval [-1, 1]"

        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


