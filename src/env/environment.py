import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium.utils import seeding
from gymnasium import spaces
from gymnasium.core import ActType, ObsType

# 10 shares per trade-share
HMAX_NORMALIZE = 10
# initial amount of money we have in our account
INITIAL_ACCOUNT_BALANCE = 100_000
# transaction fee: 1/1000 reasonable percentage
TRANSACTION_FEE_PERCENT = 0.001

class StockEnvTrade(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}
    def __init__(self, df:pd.DataFrame, day:int=0, initial:bool=True):
        self.df = df
        self.day = day
        self.initial = initial 

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
        self.cum_reward = 0
        self.actions_memory=[[1/self.stock_dim]*self.stock_dim]
    

    def _sell_stock(self, index:int, action:float) -> None:
        """
        sell stock
        Args:
        - index: stock index to sell
        - action: shares to sell
        """
        # we need to round since we cannot buy half stock
        action = np.floor(action)

        if self.state[index+self.stock_dim+1] > 0: #when owned shares > 0
            #update balance = price stock * # of stock to sell * fee
            self.state[0] += self.state[index+1]*min(abs(action),self.state[index+self.stock_dim+1]) * (1- TRANSACTION_FEE_PERCENT)
            #update owned shares
            self.state[index+self.stock_dim+1] -= min(abs(action), self.state[index+self.stock_dim+1])
        else:
            pass


    def _buy_stock(self, index:int, action:float) -> None:
        
        # we need to round since we cannot buy half stock
        action = np.floor(action)
        
        # update balance = price stock * # of stock to buy * fee
        self.state[0] -= self.state[index+1] * action * (1+ TRANSACTION_FEE_PERCENT)

        self.state[index+self.stock_dim+1] += action


    def step(self, actions: ActType) -> tuple[ObsType, float, bool, bool, dict]:
        #print("step actions", actions) not necessarily sum to 1.
        #note, the entering actions must be between -1 and 1!!!
        #observation,reward,terminated,truncated,info
        #https://gymnasium.farama.org/api/env/#gymnasium.Env.step
        self.terminal = self.day >= len(self.df.index.unique())-1
        if self.terminal:
            # print("REWARD", self.reward)
            # print("cumulative reward,", self.cum_reward)
            #return self.state, self.reward, self.terminal, False, {}
            return self.state, self.reward, self.terminal, False, {"episodic_return": self.cum_reward}

        else:
            actions *= HMAX_NORMALIZE
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
            
            #total asset = cash + stock_price_vector * stock_share_vector
            end_total_asset = self.state[0] + \
                            sum(np.array(self.state[1:(self.stock_dim+1)])*np.array(self.state[(self.stock_dim+1):(self.stock_dim*2+1)]))
            
            self.reward = end_total_asset - begin_total_asset   

            weights = self.normalization(np.array(self.state[(self.stock_dim+1):(self.stock_dim*2+1)]))

            self.actions_memory.append(weights.tolist())

            #document cumulative reward
            self.cum_reward += (end_total_asset - begin_total_asset)

        return self.state, self.reward, self.terminal, False, {}


    def reset(self, seed = None, options = None) -> tuple[ObsType, dict]:  
        #super().reset(seed=seed)
        self.day = 0
        self.data = self.df.iloc[self.day,:]
        self.terminal = False 

        # memorize all the total balance change
        self.actions_memory=[[1/self.stock_dim]*self.stock_dim]

        #initiate state
        self.state = [INITIAL_ACCOUNT_BALANCE] + self.data.values.tolist() + [0]*self.stock_dim
        self.cum_reward = 0

        #self._seed(1)
        
        #info placeholder, not implemented!
        return self.state, {}
    
    
    
    def normalization(self, actions):
        output = actions/(np.sum(actions)+1e-15)
        return output


    def save_action_memory(self):
        return self.actions_memory
    

    def render(self, mode='human',close=False):
        return self.state
    

    def set_seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.seed = seed
        return [seed]



