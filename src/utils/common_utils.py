import pandas as pd
import numpy as np
import torch
import gymnasium as gym
from collections import namedtuple
from typing import NamedTuple

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
