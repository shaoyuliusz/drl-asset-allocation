import argparse
import os
import random
import time
from distutils.util import strtobool
from itertools import count
import pandas as pd

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from src.models.ppo.ppo import make_env, Agent
from src.env.environment import StockEnvTrade


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="ppo-implementation-details",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--save-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, the trained model will be saved.")
    parser.add_argument("--save-path", nargs="?", 
                        default=os.path.dirname(__file__), 
                        help="model save path")

    # Algorithm specific arguments
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    # parser.add_argument("--total-timesteps", type=int, default=50000,
    #     help="total timesteps of the experiments")
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--num-episodes", type=int, default=100,
        help="the number of episodes to run")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=1.0,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    # parser.add_argument("--num-minibatches", type=int, default=4,
    #     help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None, #0.015
        help="the target KL divergence threshold")
    args = parser.parse_args()
    #args.batch_size = int(args.num_envs * args.num_steps)
    #args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"models/ppo/runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    df = pd.read_csv("data/yahoo_finance_train.csv")
    #df = pd.DataFrame({'AAPL':[50, 100], "JPM": [100, 50]})
    stock_env_trade = StockEnvTrade(df)
    n_days = len(df)
    
    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_ = stock_env_trade, seed = args.seed + i) for i in range(args.num_envs)]
    )

    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    input_dims = np.array(envs.single_observation_space.shape).prod()
    output_dims = np.prod(envs.single_action_space.shape)
    agent = Agent(envs, input_dims, output_dims).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup

    obs = torch.zeros((n_days, args.num_envs) + envs.single_observation_space.shape).to(device) #torch.Size([num_steps, num_envs, 17])
    actions = torch.zeros((n_days, args.num_envs) + envs.single_action_space.shape).to(device) #torch.Size([num_steps, num_envs, 8])
    logprobs = torch.zeros((n_days, args.num_envs)).to(device)
    rewards = torch.zeros((n_days, args.num_envs)).to(device)
    dones = torch.zeros((n_days, args.num_envs)).to(device)
    values = torch.zeros((n_days, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    obs_, _ = envs.reset()
    
    next_obs = torch.Tensor(obs_).to(device) #(1, 17) shaped tensor
    next_done = torch.zeros(args.num_envs).to(device) #next_done[i] corresponds to the ith sub-environment being not done and done.
    #num_updates = args.total_timesteps // args.batch_size
   
    for episode in range(1, args.num_episodes + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (episode - 1.0) / args.num_episodes
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        # ROLLOUT PHASE
        #for step in range(0, args.num_steps):
        for step in range(n_days):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                # print("action: ",action) (8,) tensor
                # print("logprob", logprob) (8,) tensor
                # print("value", value) (1,1) tensor e.g tensor([[0.1514]])
                values[step] = value.flatten()
            
            actions[step] = action
            logprobs[step] = logprob

            # execute the game and log data.
            next_obs, reward, done, _, info = envs.step(action.cpu().numpy())
            
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            for item in info:
                if item == "final_info":
                    for obj in info[item]:
                        if "episode" in obj:
                            print(f"global_step={global_step}, episodic_return={obj['episodic_return']}") #cumulative rewards in an episode
                            writer.add_scalar("charts/episodic_return", obj["episodic_return"], global_step)
                            writer.add_scalar("charts/episodic_length", obj["episode"]["l"], global_step)
                            break
        
        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            if args.gae:
                advantages = torch.zeros_like(rewards).to(device) #torch.Size([num_steps, 1])
                lastgaelam = 0
                for t in reversed(range(n_days)):
                    if t == n_days - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)               
                for t in reversed(range(n_days)):
                    if t == n_days - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                advantages = returns - values
        #print("obs.shape", obs.shape) [1409, 1, 17]
        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        #print("envs.single_observation_space.shape",envs.single_observation_space.shape) #(17,)
        #print("b_obs shape: ", b_obs.shape) #[1409, 17]
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        #LEARNING PHASE
        # Optimizing the policy and value network
        #b_inds = np.arange(args.batch_size) #batch indices
        clipfracs = []
        for epoch in range(args.update_epochs):
            #np.random.shuffle(b_inds) #shuffle batch indices
            for start in range(0, n_days):
                #end = start + args.minibatch_size #128 mini batch per time
                #mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs, b_actions)
                logratio = newlogprob - b_logprobs

                ratio = logratio.exp()

                #check if ratio = 1 https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
                # if epoch == 0 and start == 0:
                #     print(ratio)

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages #advantage of a batch
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1) #shape=[1409]

                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns) ** 2
                    v_clipped = b_values + torch.clamp(
                        newvalue - b_values,
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef #the total loss function

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm) #Global Gradient Clipping
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("losses/total_loss", loss.item(), global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()

    print("Training time: ", (time.time()- start_time)/60, " minutes")
    print("Model's state_dict:")
    for param_tensor in agent.state_dict():
        print(param_tensor, "\t", agent.state_dict()[param_tensor].size())
   
    if args.save_model:
        print(f"start saving model to {args.save_path}")
        torch.save(agent.state_dict(), os.path.join(args.save_path,"models/ppo","torch_ppo.pt"))