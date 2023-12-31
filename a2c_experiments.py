# run nohup python3 ppo_experiments.py >/dev/null 2>&1
import json
import gymnasium as gym
import numpy as np
import pandas as pd
import itertools
import tqdm
import os
from collections import defaultdict

from src.models.a2c import A2CAgent
from src.env.environment import StockEnvTrade
from src.utils.common_utils import (dict_to_namedtuple, 
                                    replace_dict, 
                                    DRL_prediction,
                                    Agent, 
                                    make_env_test
                                    )

# Define the hyperparameters to tune
# param_grid = {
#     'learning_rate': [2.5e-5, 2.5e-4, 2.5e-3],
#     'total_timesteps': [25000, 50000, 75000],
#     'num_steps': [256, 1024],
#     'gae': [True, False],
#     'update_epochs': [5, 10],
#     'norm_adv': [True, False],
#     "num_minibatches": [32, 64],
#     "normalize_env": [True, False]
# }

param_grid = {
    'learning_rate': [2.5e-5, 2.5e-4, 2.5e-3],
    'total_timesteps': [25000, 50000, 75000],
    'num_steps': [256, 1024],
    'gae': [True, False],
    "normalize_env": [True, False],
    "anneal_lr": [True, False],
}

Keys = param_grid.keys()
PARAM_SETS = [dict(zip(Keys, values)) for values in itertools.product(*param_grid.values())]

N_RUNS = 5 #average performance over 5 runs

def main():
    with open('configs/a2c_config.json') as f:
        config_dict = json.load(f)

    result = defaultdict(dict)

    df_train = pd.read_csv("data/yahoo_finance_train.csv")
    df_test = pd.read_csv("data/yahoo_finance_val.csv")

    for i, params_dict in tqdm.tqdm(enumerate(PARAM_SETS)):
        config = replace_dict(config_dict, params_dict)
        config["init_params"]["exp_name"] = f"hyperparam_{i}"
        config_params = dict_to_namedtuple(config)
        
        for j in range(N_RUNS): #run several times and average the score
            a2c_agent = A2CAgent(data = df_train, env = StockEnvTrade, agent_model = Agent,
                                init_params = config_params.init_params, 
                                algo_params = config_params.algo_params,
                                model_num = f"run{j}"
                                )
            a2c_agent.initialize()
            a2c_agent.train()
            a2c_agent.save()

            #initialize test environment
            stock_env_trade = StockEnvTrade(df = df_test)
            test_env = gym.vector.SyncVectorEnv(
                    [make_env_test(env_ = stock_env_trade, seed = 1) for i in range(1)]
                )
            #reset test env
            test_obs, _ = test_env.reset()

            portfolio_weights_a2c = np.array(DRL_prediction(df_test, a2c_agent.agent, test_env, test_obs))
            
            return_stocks = df_test.pct_change()
            return_stocks_a2c = np.sum(return_stocks.multiply(portfolio_weights_a2c), axis=1)
            cumulative_returns_daily_drl_a2c = (1+return_stocks_a2c).cumprod()

            result[i][j] = cumulative_returns_daily_drl_a2c

            save_path = os.path.join(config_params.init_params.save_path, f"hyperparam_{i}", f"run{j}")

            cumulative_returns_daily_drl_a2c.to_csv(os.path.join(save_path, f"cum_return_summary_run{j}.csv"), index=False)

    with open('models/a2c/experiment_all_result.json', 'w') as fp:
        json.dump(result, fp)

if __name__ == "__main__":
    main()