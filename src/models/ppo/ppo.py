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

class PPOAgent:
    def __init__(self, 
                 data, 
                 env: gym.Env, 
                 agent_model: nn.Module, 
                 init_params: namedtuple, 
                 algo_params: namedtuple,
                 model_num: Union[str, int] = None
                 ) -> None:
        """
        Initialize a new PPOAgent class
        Args:
            data
            env
            agent_model
            writer
            init_params
            algo_params
            model_num
        """
        self.data = data
        self.env = env
        self.agent_model = agent_model
        self.init_params = init_params
        self.algo_params = algo_params

        self.batch_size = int(self.algo_params.num_envs * self.algo_params.num_steps) #128
        self.minibatch_size = int(self.batch_size // self.algo_params.num_minibatches) #128/4
        
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
                    # print("action: ",action) (8,) tensor
                    # print("logprob", logprob) (8,) tensor
                    # print("value", value) (1,1) tensor e.g tensor([[0.1514]])
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
            b_inds = np.arange(self.batch_size) #batch indices
            clipfracs = []
            for epoch in range(self.algo_params.update_epochs):
                np.random.shuffle(b_inds) #shuffle batch indices
                for start in range(0, self.batch_size, self.minibatch_size):
                    end = start + self.minibatch_size #128 mini batch per time
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]

                    ratio = logratio.exp()

                    #check if ratio = 1 https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
                    # if epoch == 0 and start == 0:
                    #     print(ratio)

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > self.algo_params.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds] #advantage of a batch
                    if self.algo_params.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.algo_params.clip_coef, 1 + self.algo_params.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if self.algo_params.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -self.algo_params.clip_coef,
                            self.algo_params.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - self.algo_params.ent_coef * entropy_loss + v_loss * self.algo_params.vf_coef #the total loss function

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.algo_params.max_grad_norm) #Global Gradient Clipping
                    self.optimizer.step()

                if self.algo_params.target_kl is not None:
                    if approx_kl > self.algo_params.target_kl:
                        break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            self.writer.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], global_step)
            self.writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            self.writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            self.writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            self.writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            self.writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
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

def make_env(env_: gym.Env, seed:int, use_normalization: bool=True) -> Callable[[], gym.Env]:
    def thunk():
        if use_normalization:
            #env = gym.make(gym_id)
            env = gym.wrappers.RecordEpisodeStatistics(env_)
            env = gym.wrappers.ClipAction(env)
            env = gym.wrappers.NormalizeObservation(env) #observation scaling
            env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, 0, 10)) #observation clipping
            env = gym.wrappers.NormalizeReward(env) #reward scaling
            env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10)) #reward clipping
            env.seed(seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env
        else:
            env_.seed(seed)
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
    def __init__(self, envs, input_dims, output_dims):
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
            layer_init(nn.Linear(input_dims, 64)), #np.array(envs.single_observation_space.shape).prod()
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(input_dims, 64)), #np.array(envs.single_observation_space.shape).prod()
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, output_dims), std=0.01), #np.prod(envs.single_action_space.shape)=8
        )
        #state independent standard deviations
        self.actor_logstd = nn.Parameter(torch.zeros(1, output_dims)) #np.prod(envs.single_action_space.shape)
        self.envs = envs


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
        #assert isinstance(self.envs.single_action_space, spaces.Box) == True     
        #print("self.actor_mean(x)", self.actor_mean(x).shape) [1,8]
        action_mean = self.actor_mean(x)
        #action_mean = torch.reshape(self.actor_mean(x), (1,-1))
        #print("action_mean shape", action_mean.shape) [1,8]
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std) #Creates a normal distribution parameterized by action_mean and action_std

        if action is None:
            action = probs.sample()
            # Actions could be on arbitrary scale, so clip the actions to avoid
            # out of bound error (e.g. if sampling from a Gaussian distribution)
        
        #clip action, see https://ai.stackexchange.com/questions/28572/how-to-define-a-continuous-action-distribution-with-a-specific-range-for-reinfor
        #action = np.clip(action, self.envs.single_action_space.low, self.envs.single_action_space.high)
        #assert torch.all((action >= -1) & (action <= 1)), "Not all elements are in the interval [-1, 1]"

        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


