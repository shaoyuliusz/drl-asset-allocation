import os
import json
import gymnasium as gym
import numpy as np
import pandas as pd
import math
import torch
import itertools
import tqdm
import sys

from collections import namedtuple, defaultdict

from src.models.ppo.ppo import Agent, PPOAgent
from src.env.environment import StockEnvTrade

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

with open('configs/ppo/basic_config.json') as f:
    config_dict = json.load(f)

# Define the hyperparameters to tune
param_grid = {
    'learning_rate': [2.5e-5, 2.5e-4, 2.5e-3],
    'total_timesteps': [10000, 30000, 50000],
    'num_steps': [64, 128, 256],
    'gae': [True, False],
    'update_epochs': [5, 10],
    'norm_adv': [True, False],
    'target_kl': [None, 0.015]
}
Keys = param_grid.keys()
PARAM_SETS = [dict(zip(Keys, values)) for values in itertools.product(*param_grid.values())]

N_RUNS = 5

def dict_to_namedtuple(dictionary, name='Params'):
    for key, value in dictionary.items():
        if isinstance(value, dict):
            # Recursively convert nested dictionaries to named tuples
            dictionary[key] = dict_to_namedtuple(value, f'{name}_{key.capitalize()}')
    return namedtuple(name, dictionary.keys())(**dictionary)


def replace_dict(dict1, dict2) -> None:
    # Replace values in dict1 with values from dict2, return a new dict object
    dict_ = dict1.copy()
    for key in dict_:
        if key in dict2:
            dict_[key] = dict2[key]
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
            env = env.env_fns[0]()
            actions_memory = env.save_action_memory()
    
    return actions_memory #this must be (days X n_stocks)

def make_env_test(env_: gym.Env, seed:int):
    def thunk():
        #env = gym.make(gym_id)
        #env = gym.wrappers.RecordEpisodeStatistics(env_)
        env = gym.wrappers.ClipAction(env_)
        env = gym.wrappers.NormalizeObservation(env) #observation scaling
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10)) #observation clipping
        env = gym.wrappers.NormalizeReward(env) #reward scaling
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10)) #reward clipping
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk


def main():
    result = defaultdict(dict)

    df_train = pd.read_csv("data/yahoo_finance_train.csv")
    df_test = pd.read_csv("data/yahoo_finance_test.csv")

    for i, params_dict in tqdm.tqdm(enumerate(PARAM_SETS)):
        config = replace_dict(config_dict, params_dict)
        config_params = dict_to_namedtuple(config)
        
        for j in range(N_RUNS): #run several times and average the score
            ppo_agent = PPOAgent(data = df_train, env = StockEnvTrade, agent_model = Agent,
                                init_params = config_params.init_params, 
                                algo_params = config_params.algo_params,
                                model_num = f"{i}_{j}"
                                )
            ppo_agent.initialize()
            ppo_agent.train()

            #initialize test environment
            stock_env_trade = StockEnvTrade(df = df_test)
            test_env = gym.vector.SyncVectorEnv(
                    [make_env_test(env_ = stock_env_trade, seed = 1) for i in range(1)]
                )
            #reset test env
            test_obs, _ = test_env.reset()

            portfolio_weights_ppo = np.array(DRL_prediction(df_test, ppo_agent.agent, test_env, test_obs))
            
            assert math.isclose(np.sum(portfolio_weights_ppo), len(df_test)) == True, "portfolio sum must equal to number of trade days"

            return_stocks = df_test.pct_change()
            return_stocks_ppo = np.sum(return_stocks.multiply(portfolio_weights_ppo), axis=1)
            cumulative_returns_daily_drl_ppo = (1+return_stocks_ppo).cumprod()

            result[i][j] = cumulative_returns_daily_drl_ppo

    with open('models/ppo/experiment_result.json', 'w') as fp:
        json.dump(result, fp)

if __name__ == "__main__":
    main()