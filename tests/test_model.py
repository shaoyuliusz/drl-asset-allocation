import pytest
import numpy as np
import math
import gymnasium as gym

from src.utils.common_utils import DRL_prediction, Agent, make_env_test
from src.env.environment import StockEnvTrade

@pytest.fixture
def input_env(input_stock_data):
    test_env = StockEnvTrade(input_stock_data)
    test_env = gym.vector.SyncVectorEnv(
                [make_env_test(env_ = test_env, seed = 1) for i in range(1)]
            )
    return test_env

@pytest.fixture
def input_model(input_env):
    #initialize the model
    model = Agent(envs=input_env, 
                input_dims=np.array(input_env.observation_space.shape).prod(),
                output_dims=np.prod(input_env.action_space.shape))
    return model


#Post-train tests
def test_drl_model(input_stock_data, input_env, input_model):
    """
    test the generated portfolio weights make sense
    """

    test_obs, _ = input_env.reset()
    portfolio_weights_ppo = np.array(DRL_prediction(input_stock_data, input_model, input_env, test_obs))
    
    assert portfolio_weights_ppo.shape[0] == input_stock_data.shape[0], "number of trade days in portfolio must equal to number of trade days in stock data"
    assert portfolio_weights_ppo.shape[1] == input_stock_data.shape[1], "number of portfolio stocks must equal to number in the stock data"
    for day in range(portfolio_weights_ppo.shape[0]):
        assert math.isclose(portfolio_weights_ppo[day].sum(), 1), f"for day {day}, portfolio weights must sum to 1."
    assert math.isclose(np.sum(portfolio_weights_ppo), len(input_stock_data)) == True, "portfolio sum must equal to number of trade days"
    