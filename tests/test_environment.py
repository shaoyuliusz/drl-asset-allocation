import pytest
import gymnasium as gym
import pandas as pd
import numpy as np
import math
from src.env.environment import StockEnvTrade
from src.models.ppo.ppo import make_env

SEED = 123
TEST_DATA_FILE = "data/yahoo_finance_test.csv"
NUM_STEPS = 100

@pytest.fixture
def test_envs():
    test_data = pd.read_csv(TEST_DATA_FILE)
    test_env_1 = StockEnvTrade(test_data)
    test_env_2 = StockEnvTrade(test_data)
    return [test_env_1, test_env_2]

@pytest.fixture
def test_env():
    test_data = pd.read_csv(TEST_DATA_FILE)
    test_env = StockEnvTrade(test_data)
    return test_env


def assert_equals(a, b, prefix=None):
    """Assert equality of data structures `a` and `b`.

    Args:
        a: first data structure
        b: second data structure
        prefix: prefix for failed assertion message for types and dicts
    """
    assert type(a) == type(b), f"{prefix}Differing types: {a} and {b}"
    if isinstance(a, dict):
        assert list(a.keys()) == list(b.keys()), f"{prefix}Key sets differ: {a} and {b}"

        for k in a.keys():
            v_a = a[k]
            v_b = b[k]
            assert_equals(v_a, v_b)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, tuple):
        for elem_from_a, elem_from_b in zip(a, b):
            assert_equals(elem_from_a, elem_from_b)
    else:
        assert a == b


def test_make_env(test_env: gym.Env) -> None:
    test_env_func = make_env(test_env, seed = SEED)
    assert callable(test_env_func)
    assert isinstance(test_env_func(), gym.Env)
    test_env.close()


def test_actions_memory(test_env: gym.Env) -> None:
    """Run environment and test if actions_memory works properly"""
    for time_step in range(NUM_STEPS):
        action = test_env.action_space.sample()
        obs, rew, terminated, truncated, info = test_env.step(action)
    
    actions_memory = test_env.save_action_memory()
    for act in actions_memory:
        assert len(act) == test_env.action_space.shape[0] #number of actions = 8
        assert math.isclose(np.sum(act), 1) #if normalized actions (stock share weights) sum to 1

    test_env.close()


def test_env_determinism_rollout(test_envs):
    """Run a rollout with two environments and assert equality.

    This test run a rollout of NUM_STEPS steps with two environments
    initialized with the same seed and assert that:

    - observation after first reset are the same
    - same actions are sampled by the two envs
    - observations are contained in the observation space
    - obs, rew, done and info are equals between the two envs
    """
    env_1, env_2 = test_envs
    env_1.seed(SEED)
    env_2.seed(SEED)
    initial_obs_1, initial_info_1 = env_1.reset(seed=SEED)
    initial_obs_2, initial_info_2 = env_2.reset(seed=SEED)
    assert_equals(initial_obs_1, initial_obs_2)

    env_1.action_space.seed(SEED)
    env_2.action_space.seed(SEED)

    for time_step in range(NUM_STEPS):
        action = env_1.action_space.sample()
        assert env_1.action_space.contains(action)

        obs_1, rew_1, terminated_1, truncated_1, info_1 = env_1.step(action)
        obs_2, rew_2, terminated_2, truncated_2, info_2 = env_2.step(action)
        print(obs_1)
        print(obs_2)
        
        print(rew_1)
        print(rew_2)
        assert_equals(obs_1, obs_2, f"[{time_step}] ")
        assert env_1.observation_space.contains(
            obs_1
        )  # obs_2 verified by previous assertion

        assert rew_1 == rew_2, f"[{time_step}] reward 1={rew_1}, reward 2={rew_2}"
        assert (
            terminated_1 == terminated_2
        ), f"[{time_step}] done 1={terminated_1}, done 2={terminated_2}"
        assert (
            truncated_1 == truncated_2
        ), f"[{time_step}] done 1={truncated_1}, done 2={truncated_2}"
        assert_equals(info_1, info_2, f"[{time_step}] ")

        if (
            terminated_1 or truncated_1
        ):  # terminated_2, truncated_2 verified by previous assertion
            env_1.reset(seed=SEED)
            env_2.reset(seed=SEED)

    env_1.close()
    env_2.close()
