# Deep Reinforcement Learning in Markowitz Asset Allocation

This repository contains code and introduction to training deep reinforcement agents for asset allocation. In particular, it contains our implementation of Actor-Critic (A2C) and PPO algorithms, with code level optimizations and experiments. 

## Overview

In this project, we use deep reinforcement learning to address asset allocation problem in finance. We demonstrate this on daily data for a number of stocks in the US equities with daily re-balancing. In particular, we apply classic reinforcement learning algorithms, such as actor-critic (A2C) and proximal policy optimization (PPO) algorithms to dynamically optimize asset allocation to maximize stock returns. Through objective evaluation of performance and careful ablation studies, we aim to dive deep to the algorithm implementation details, evaluate their performance against classical optimization methodologies： Mean-Variance and Sharpe-CVaR optimization. We horizontally compared the results in dimensions of profit derivation, risk maximum draw-down and volatility to provide empirical evidence on reinforcement learning's feasibility as a robust tools in contributing to asset allocation.

## Project Structure
1 - The main results in the paper are in the folder `notebooks`.
* `ablation-studies.ipynb` reports our ablation study results.
* `plot_best_models.ipynb` compares our reinforcement learning agents with baseline models in test data.

2 - The default configuration files for training PPO and A2C agents are in `experiment_configs` folder. To perform grid search for model hyperparameters, please edit the PARAM_GRID variable in ppo_experiments.py, and make a new `experiment` folder to train new agents. For example,
```python
PARAM_GRID = {
    'learning_rate': [2.5e-3],
    'total_timesteps': [100000, 150000],
    'num_steps': [1024],
    'gae': [True, False],
    'norm_adv': [True, False],
    "normalize_env": [True],
    "anneal_lr": [True, False],
}
```

3 - `src/models` folder contains our A2C in `a2c.py` and PPO models in `ppo.py`, `src/env` folder sets up our custom stock trading environment, `example_models` folder provides such example trained models.

## References

The PPO implementation is based on the open-source code in [**Huang, et al., "The 37 Implementation Details of Proximal Policy Optimization ICLR Blog Track, 2022"**](https://github.com/vwxyzjn/ppo-implementation-details/blob/main/ppo.py).
