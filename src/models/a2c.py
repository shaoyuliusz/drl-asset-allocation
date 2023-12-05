import os
import json
import gymnasium as gym
import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Callable
from torch.distributions.normal import Normal
from typing import Union
from torch.utils.tensorboard import SummaryWriter
from collections import namedtuple

from utils.common_utils import Agent, make_env

class A2CAgent:
    def __init__(self, 
                 data, 
                 env: gym.Env, 
                 agent_model: nn.Module, 
                 init_params: namedtuple, 
                 algo_params: namedtuple,
                 model_num: Union[str, int] = None
                 ) -> None:
        """
        Initialize a new A2CAgent class
        Args:
            data
            env
            agent_model
            writer
            init_params:
            algo_params: a namedtuple of algorithm parameters
            model_num:
        """
        self.data = data
        self.env = env
        self.agent_model = agent_model
        self.init_params = init_params
        self.algo_params = algo_params

        self.batch_size = int(self.algo_params.num_envs * self.algo_params.num_steps) #128
        
        if not model_num:
            model_num = np.random.randint(10000)
            while os.path.exists(save_path):
                model_num = np.random.randint(10000)
        
        save_path = os.path.join(self.init_params.save_path, self.init_params.exp_name)
        self.model_num = model_num
        self.save_path = save_path

        hyper_params = self.init_params._asdict() | self.algo_params._asdict()
        writer = SummaryWriter(os.path.join(save_path, f"{self.model_num}"))
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in hyper_params.items()])),
        )
        self.writer = writer
    
    def initialize(self):
        # fix training
        if self.init_params.seed:
            seed = self.init_params.seed
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = self.init_params.torch_deterministic
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.init_params.cuda else "cpu")

        stock_env_trade = self.env(self.data)
        self.model_envs = gym.vector.SyncVectorEnv(
            [make_env(env_ = stock_env_trade, seed = i, use_normalization=self.algo_params.normalize_env) for i in range(self.algo_params.num_envs)]
            )
        
        assert isinstance(self.model_envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

        input_dims = np.array(self.model_envs.single_observation_space.shape).prod()
        output_dims = np.prod(self.model_envs.single_action_space.shape)
        
        self.agent = self.agent_model(self.model_envs, input_dims, output_dims).to(self.device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.algo_params.learning_rate, eps=1e-5)

        # ALGO Logic: Storage setup
        num_steps = self.algo_params.num_steps
        num_envs = self.algo_params.num_envs

        self.obs = torch.zeros((num_steps, num_envs) + self.model_envs.single_observation_space.shape).to(self.device) #torch.Size([num_steps, num_envs, 17])
        self.actions = torch.zeros((num_steps, num_envs) + self.model_envs.single_action_space.shape).to(self.device) #torch.Size([num_steps, num_envs, 8])
        self.logprobs = torch.zeros((num_steps, num_envs)).to(self.device)
        self.rewards = torch.zeros((num_steps, num_envs)).to(self.device)
        self.dones = torch.zeros((num_steps, num_envs)).to(self.device)
        self.values = torch.zeros((num_steps, num_envs)).to(self.device)

    def train(self):
        """
        train the agent
        """
        # TRY NOT TO MODIFY: start the game
        global_step = 0
        start_time = time.time()
        obs_, _ = self.model_envs.reset()
        
        next_obs = torch.Tensor(obs_).to(self.device) #(1, 17) shaped tensor
        next_done = torch.zeros(self.algo_params.num_envs).to(self.device) #next_done[i] corresponds to the ith sub-environment being not done and done.
        num_updates = self.algo_params.total_timesteps // self.batch_size
    
        for update in range(1, num_updates + 1):
            # Annealing the rate if instructed to do so.
            if self.algo_params.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                lrnow = frac * self.algo_params.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow
            
            # ROLLOUT PHASE
            for step in range(0, self.algo_params.num_steps):
                global_step += 1 * self.algo_params.num_envs
                self.obs[step] = next_obs
                self.dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(next_obs)
                    self.values[step] = value.flatten()
                
                self.actions[step] = action
                self.logprobs[step] = logprob

                # execute the game and log data.
                next_obs, reward, done, _, info = self.model_envs.step(action.cpu().numpy())

                self.rewards[step] = torch.Tensor(reward).to(self.device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(self.device), torch.Tensor(done).to(self.device)

                for item in info:
                    if item == "final_info":
                        for obj in info[item]:
                            if "episode" in obj:
                                print(f"global_step={global_step}, episodic_return={obj['episodic_return']}, episodic_length={obj['episode']['l']}") #cumulative rewards in an episode
                                self.writer.add_scalar("charts/episodic_return", obj["episodic_return"], global_step)
                                self.writer.add_scalar("charts/episodic_length", obj["episode"]["l"], global_step)
                                break
            
            # bootstrap value if not done
            with torch.no_grad():
                next_value = self.agent.get_value(next_obs).reshape(1, -1)
                if self.algo_params.gae:
                    advantages = torch.zeros_like(self.rewards).to(self.device) #torch.Size([num_steps, 1])
                    lastgaelam = 0
                    for t in reversed(range(self.algo_params.num_steps)):
                        if t == self.algo_params.num_steps - 1:
                            nextnonterminal = 1.0 - next_done
                            nextvalues = next_value
                        else:
                            nextnonterminal = 1.0 - self.dones[t + 1]
                            nextvalues = self.values[t + 1]
                        delta = self.rewards[t] + self.algo_params.gamma * nextvalues * nextnonterminal - self.values[t]
                        advantages[t] = lastgaelam = delta + self.algo_params.gamma * self.algo_params.gae_lambda * nextnonterminal * lastgaelam
                    returns = advantages + self.values
                else:
                    returns = torch.zeros_like(self.rewards).to(self.device)    
                    for t in reversed(range(self.algo_params.num_steps)): #reward-to-go https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html#implementing-reward-to-go-policy-gradient
                        if t == self.algo_params.num_steps - 1:
                            nextnonterminal = 1.0 - next_done
                            next_return = next_value
                        else:
                            nextnonterminal = 1.0 - self.dones[t + 1]
                            next_return = returns[t + 1]
                        returns[t] = self.rewards[t] + self.algo_params.gamma * nextnonterminal * next_return
                    advantages = returns - self.values

            # flatten the batch
            b_obs = self.obs.reshape((-1,) + self.model_envs.single_observation_space.shape)
            b_logprobs = self.logprobs.reshape(-1)
            b_actions = self.actions.reshape((-1,) + self.model_envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = self.values.reshape(-1)

            #LEARNING PHASE
            # Optimizing the policy and value network
            for step in range(0, self.batch_size):
                _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_obs, b_actions)

                # Policy loss
                pg_loss = -(newlogprob*b_advantages).mean()

                # Value loss
                v_loss = 0.5 * ((newvalue - b_returns) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.algo_params.ent_coef * entropy_loss + v_loss * self.algo_params.vf_coef #the total loss function

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.algo_params.max_grad_norm) #Global Gradient Clipping
                self.optimizer.step()

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            self.writer.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], global_step)
            self.writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            self.writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            self.writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            self.writer.add_scalar("losses/explained_variance", explained_var, global_step)
            self.writer.add_scalar("losses/total_loss", loss.item(), global_step)
            self.writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        self.model_envs.close()
        self.writer.close()

        print("Training time: ", (time.time()- start_time)/60, " minutes")
        print(f"start saving model to {self.save_path}")

    def save(self):
        #save model
        torch.save(self.agent.state_dict(), os.path.join(self.save_path, self.model_num, "torch_ppo.pt"))

        #save hyperparameters
        hyper_params = self.init_params._asdict() | self.algo_params._asdict()
        with open(os.path.join(self.save_path, 'hyper_params.json'), 'w') as fp:
            json.dump(hyper_params, fp)