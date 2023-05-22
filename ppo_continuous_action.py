import argparse,os,time,random
import torch
from distutils.util import strtobool

import numpy as np
# agent setup
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
# from pyuac import main_requires_admin
import gymnasium as gym
from gymnasium.experimental.wrappers.rendering import RecordVideoV0 as RecordVideo

def parse_args():
    '''
    setup some common variables
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")    
    parser.add_argument("--env-id", type=str, default="HalfCheetah-v4",
        help="the id of the gym environment")    
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=1000000,
        help="total timesteps of the experiments")
    # two gpu related variables
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    # Weight and Biases setup
    parser.add_argument("--track", action='store_true', 
        help="if toggled, this experiment will be tracked with Weights and Biases") # default is false
    parser.add_argument("--wandb-project-name", type=str, default="DeepRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", action='store_true',
        help="weather to capture videos of the agent performances (check out `videos` folder)")    

    # Algorithm specific arguments
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=2048,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    # discount for all advantage function
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma") 
    # extra for GAE
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=32,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=10,
        help="the K epochs to update the policy") 
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.0,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")    
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
                   
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    # ppo break the batch data into mini batches for training
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args

def make_env(env_id, idx, capture_video, run_name, gamma):
    '''
    Environment Preprocessing
    '''
    def thunk():
        if capture_video:
            env = gym.make(env_id, render_mode="rgb_array")
        else:
            env = gym.make(env_id)
        # Wrapper--keep track of cumulative rewards and episode lengths.
        # At the end of an episode, the statistics of the episode will be added to info. 
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        # records the episode statistics and videos
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = RecordVideo(env, f"videos/{run_name}")
                # env = gym.wrappers.RecordVideoV0(env, f"videos/{run_name}",episode_trigger=lambda t:t%100==0)

        # pre-processes the environment
        # action clipping to valid range and storage (the sampling of the Gaussian distribution has no boundaries)
        # clips the action to the lower and upper bounds of the action space
        env = gym.wrappers.ClipAction(env)
        # Normalization of Observation (standardization, have zero mean and unit variance)
        # The raw observation was normalized by subtracting its running mean and divided by its variance.
        env = gym.wrappers.NormalizeObservation(env)
        # Observation Clipping 
        #  it might be helpful in an environment with a wide range of observation
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        # Reward Scaling, applies a certain discount-based scaling scheme,
        # rewards are divided by the standard deviation of a rolling discounted sum of the rewards 
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        # Reward Clipping
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    '''
    layer init function
    orthogonal initialization on the weight
    constant initialization on the bias 
    '''
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    '''
    setup the agent class
    '''
    def __init__(self, envs):
        super(Agent, self).__init__()
        # setup critic network
        self.critic = nn.Sequential(
            # input
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            # output lineaer leyer ues 1 as standard deviation
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        # setup actor network
        # Continuous actions via normal distributions
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            # continuous action space, need product
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )  
        # State-independent log standard deviation 
        # state-indepentdent means it just learnable neural network parameters, dont take any input
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))
   

    def get_value(self, x):
        '''
        estimates the value function
        :param: obs
        :return: the state-value
        '''
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        '''
        updates the policy distribution
        :param: obs in each experiment
        :return: action in each experiment,  the logarithm of the probability mass function of the normal distribution
                 Entropy of probability, the state-value
        '''
        # Gaussian distribution with full covariance is used to represent the policy
        # which means that the action selection for each dimension is performed independently.
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        # sum according to  logarithm addition rule 
        # summation of log probability of taking the action components is same as 
        # log producnt of the probability of taking the action components, which is same as
        # log probability of take the action at the current time step
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

if __name__ == "__main__":
    args = parse_args()
    # f indicates that it is a formatted string literal (contains expressions inside curly braces)
    run_name = f"{args.env_id}_{args.exp_name}_{args.seed}_{int(time.time())}"

    if args.track:
        import wandb
        # wandb.tensorboard.patch( root_logdir=f"runs/{run_name}/")
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )    
    

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # set gpu or cpu device
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup (Vectorized Environments)
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    # envs=envs.reset(seed=args.seed)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # initial agent
    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    # the batch_size is num_envs*num_steps (the size of rollout data)
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    # track the number of environment steps
    global_step = 0
    # calculate frames per second
    start_time = time.time()
    # store initial observation
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    # store initial termination condition
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    video_filenames = set()

    # iterate all episode
    # each update corrspond to one training loop
    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        
        # policy rollout is a inner loop
        for step in range(0, args.num_steps):
            # count all parallel game environments
            global_step += 1 * args.num_envs
            # store single setp's tensor
            obs[step] = next_obs
            dones[step] = next_done 

            # ALGO LOGIC: action logic
            # temporarily disables gradient computation, PyTorch operations won't track gradients
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob      

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            # flatten a tensor into a 1D vector
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            # Only print when at least 1 env is done
            if "final_info" not in infos:
                continue

            for info in infos["final_info"]:
                # Skip the envs that are not done
                if info is None:
                    continue
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # bootstrap value if exeripment not done
        # calculate the advantage values
        with torch.no_grad():
            #  if a sub-environment is not terminated nor truncated
            # PPO estimates the value of the next state in this sub-environment as the value target
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch 
        # parameter is tuple (-1,envs.single_observation_space.shape)
        # -1 means flatten 
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        # requie all the indices of the batch 
        b_inds = np.arange(args.batch_size)
        clipfracs = []

        # iterate all update epoch
        for epoch in range(args.update_epochs):
            # shuffle batch indices
            np.random.shuffle(b_inds)
            # iterate minibatch
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                # mini batch indics
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                # calculate the ratio
                # taking the exponential of a difference between two log probabilities is a more numerically stable way of computing the ratio between two probabilities 
                # avoid numerical underflow or overflow issues.
                logratio = newlogprob -  b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approximate Kullback–Leibler divergence 
                    old_approx_kl = (-logratio).mean()
                    # extimator K_3 http://joschu.net/blog/kl-approx.html
                    approx_kl = ((ratio - 1) - logratio).mean() 

                    # the fraction of the training data that triggered the clipped objective.
                    # measure how often a clipped objective is actually triggered
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()] 

                mb_advantages = b_advantages[mb_inds]
                # advantages normalization
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                # here maximun the negative equival the minimum in paper 
                pg_loss1 = -mb_advantages * ratio
                # Clamps all elements in input into the range [ min, max ]
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss, PPO clips the value function like the PPO’s clipped surrogate objective
                # recent research find value function loss clipping has no help even hurt performance
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    # normal value loss method
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                
                # entropy loss (measure the chaos and action probability distribution)
                entropy_loss = entropy.mean()
                # overall loss (multipy coefficient)
                # minimize the policy and value loss but maximize the entropy loss (encourage exploration )
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                # implement the back propagation
                optimizer.zero_grad()
                # compute a backward pass on the loss function
                # compute the gradients of the loss function
                loss.backward()
                # Global Gradient Clipping
                # PPO rescales the gradients of the policy and value network
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                # take a step with the optimizer
                optimizer.step()

            # Early Stopping of the policy optimizations 
            if args.target_kl is not None:
                # KL divergence exceeds a preset threshold
                # the updates to the policy weights are preemptively stopped.
                if approx_kl > args.target_kl:
                    break

        # prediction--value true--Q value
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        # Compute the variance along the specified axis.
        var_y = np.var(y_true)
        # explained variance (EV->1, means y_true(Q) well approximated)
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
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        if args.track and args.capture_video:
            for filename in os.listdir(f"videos/{run_name}"):
                if filename not in video_filenames and filename.endswith(".mp4"):
                    wandb.log({f"videos": wandb.Video(f"videos/{run_name}/{filename}")})
                    video_filenames.add(filename)

    envs.close()
    writer.close()