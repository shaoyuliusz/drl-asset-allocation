import numpy as np
import math
import time

import gymnasium as gym
from gymnasium.utils import seeding
from gymnasium import spaces

# 10 shares per trade-share
HMAX_NORMALIZE = 10
# initial amount of money we have in our account
INITIAL_ACCOUNT_BALANCE= 0
# transaction fee: 1/1000 reasonable percentage
TRANSACTION_FEE_PERCENT = 0.001

class StockEnvTrade(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}
    def __init__(self, size):
        
        # Total number of stocks in our portfolio
        self.stock_dim = self.df.shape[1]

        # Action Space continuous
        # action_space normalization and shape is self.stock_dim
        self.action_space = spaces.Box(low = -1, high = 1, shape = (self.stock_dim,)) 

        # State Space continuous
        # Shape = 1+8+8: [Current Balance]+[stock prices]+[owned shares] 
        self.observation_space = spaces.Box(low = 0, high = np.inf, shape = (2*self.stock_dim+1,))
        
        # load data from a pandas dataframe
        self.data = self.df.iloc[self.day,:]
        self.terminal = False
        
        # initalize state [Current Balance]+[prices]+[owned shares] 
        self.state = [INITIAL_ACCOUNT_BALANCE] + self.data.values.tolist() + [0]*self.stock_dim
        
        # initialize reward
        self.reward = 0

        # memorize all the total balance change
        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        self.actions_memory=[[1/self.stock_dim]*self.stock_dim]
        
        self._seed(0) 

    def _sell_stock(self, index:int, action:float) -> None:

        # we need to round since we cannot buy half stock
        action = np.floor(action)

        #when owned shares > 0
        if self.state[index+self.stock_dim+1] > 0:
            #update balance = price stock * # of stock to sell * fee
            self.state[0] += self.state[index+1]*min(abs(action),self.state[index+self.stock_dim+1]) * (1- TRANSACTION_FEE_PERCENT)

            self.state[index+self.stock_dim+1] -= min(abs(action), self.state[index+self.stock_dim+1])
        else:
            pass

    def _buy_stock(self, index:int, action:float) -> None:
        
        # we need to round since we cannot buy half stock
        action = np.floor(action)
        
        # update balance = price stock * # of stock to buy * fee
        self.state[0] -= self.state[index+1] * action * (1+ TRANSACTION_FEE_PERCENT)

        self.state[index+self.stock_dim+1] += action

    def step(self, actions):
        self.terminal = self.day >= len(self.df.index.unique())-1
        
        if self.terminal:
            return self.state, self.reward, self.terminal,{}

        else:       
            actions = actions * HMAX_NORMALIZE
            
            begin_total_asset = self.state[0]+ \
            sum(np.array(self.state[1:(self.stock_dim+1)])*np.array(self.state[(self.stock_dim+1):(self.stock_dim*2+1)]))
            argsort_actions = np.argsort(actions)
            
            sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                self._sell_stock(index, actions[index])
        
            for index in buy_index:
                self._buy_stock(index, actions[index])

            self.day += 1
            self.data = self.df.iloc[self.day,:]

            #load next state i.e. the new value of the stocks
            self.state =  [self.state[0]] + self.data.values.tolist() + \
                            list(self.state[(self.stock_dim+1):(self.stock_dim*2+1)])

            end_total_asset = self.state[0] + \
                            sum(np.array(self.state[1:(self.stock_dim+1)])*np.array(self.state[(self.stock_dim+1):(self.stock_dim*2+1)]))
        
            self.reward = end_total_asset - begin_total_asset            
            weights = self.normalization(np.array(self.state[(self.stock_dim+1):(self.stock_dim*2+1)]))

            self.actions_memory.append(weights.tolist())
            self.reward = self.reward

        return self.state, self.reward, self.terminal, {}

    def reset(self):  
        
        self.day = 0
        self.data = self.df.iloc[self.day,:]
        self.terminal = False 

        # memorize all the total balance change
        self.actions_memory=[[1/self.stock_dim]*self.stock_dim]

        #initiate state
        self.state = [INITIAL_ACCOUNT_BALANCE] + self.data.values.tolist() + [0]*self.stock_dim
        
        self._seed(0) 

        return self.state
    
    def normalization(self, actions):
        output = actions/(np.sum(actions)+1e-15)
        return output

    def save_action_memory(self):
        return self.actions_memory
    
    def render(self, mode='human',close=False):
        return self.state
    
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    