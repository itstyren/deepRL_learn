import argparse,os,time,random
import torch
import numpy as np
# agent setup
import torch.nn as nn
import torch.optim as optim
# Base class that represent a buffer (rollout or replay)
from stable_baselines3.common.buffers import ReplayBuffer
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
from distutils.util import strtobool


def parse_args():
    '''
    setup some common variables
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    # two gpu related variables
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    # Wnadb setup
    parser.add_argument("--track", action='store_true', 
        help="if toggled, this experiment will be tracked with Weights and Biases") # default is false
    parser.add_argument("--save-model",  action='store_true',
        help="whether to save model into the `runs/{run_name}` folder")
    parser.add_argument("--wandb-project-name", type=str, default="DeepRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", action='store_true',
        help="weather to capture videos of the agent performances (check out `videos` folder)")    

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="CartPole-v1",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=500000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--buffer-size", type=int, default=10000,
        help="the replay memory buffer size")
    # discount for all advantage function
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=1.,
        help="the target network update rate")
    parser.add_argument("--target-network-frequency", type=int, default=500,
        help="the timesteps it takes to update the target network")
    parser.add_argument("--batch-size", type=int, default=128,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--start-e", type=float, default=1,
        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, default=0.05,
        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, default=0.5,
        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument("--learning-starts", type=int, default=10000,
        help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, default=10,
        help="the frequency of training")
    
    args = parser.parse_args()
    # fmt: on
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"

    return args


def make_env(env_id, seed, idx, capture_video, run_name):
    '''
    define gym environment
    '''
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        # Wrapper--keep track of cumulative rewards and episode lengths.
        # At the end of an episode, the statistics of the episode will be added to info. 
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)

        return env

    return thunk


class QNetwork(nn.Module):
    '''
    creat the Q-network
    '''
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.single_action_space.n),
        )

    def forward(self, x):
        '''
        implements the operations on input data in the forward method.
        '''
        return self.network(x)

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    '''
    calculate exploration epsilon for current time step

    :param start_e: the starting epsilon for exploration
    :param end_e: the ending epsilon for exploration
    :param duration: exploration_fraction * total_timestep
    :param t: current time step
    '''
    # calculate the slope of exploration fraction change 
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

if __name__ == "__main__":
    import stable_baselines3 as sb3

    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
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

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup (Vectorized Environments)
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    # Q-network
    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    # target-network
    target_network = QNetwork(envs).to(device)
    # copy from the Q-network
    target_network.load_state_dict(q_network.state_dict())

    # create the experience replay buffer
    # buffer_size, obs_size, action_size, device, 
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # calculate curret epsilon for exploration according to slope and current timestep
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        # select action according to epsilon-greed policy
        # actuall output [one element]
        if random.random() < epsilon:
            # generates an array of actions by randomly sampling from the action space of an environment.
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            # returns the indices of the maximum values along dimension 1 
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        #execute the game and log data.
        next_obs, rewards, terminated, truncated, infos = envs.step(actions)

        # Record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                # Skip the envs that are not done
                if "episode" not in info:
                    continue 
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                writer.add_scalar("charts/epsilon", epsilon, global_step)

        real_next_obs = next_obs.copy()
        for idx, d in enumerate(truncated):
            if d:
                real_next_obs[idx] = infos["final_observation"][idx]
        # add data to replay buffer
        rb.add(obs, real_next_obs, actions, rewards, terminated, infos)

        # CRUCIAL step easy to overlook
        obs = next_obs
        
        # training process (after learning_starts timesetp, then the training will be starteds)
        if global_step > args.learning_starts:
            # not every step will implement training
            if global_step % args.train_frequency == 0:
                # select a minibatch from replay buffer
                data = rb.sample(args.batch_size)  
                # calculate the loss       
                with torch.no_grad():
                    # Returns a namedtuple (values, indices) 
                    target_max, _ = target_network(data.next_observations).max(dim=1)
                    # approximate target value
                    td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                # Gathers values along an axis specified by dim
                # here dim=1, means selcet the action according column index
                # squeeze() Returns a tensor with all specified dimensions of input of size 1 removed.
                old_val = q_network(data.observations).gather(1, data.actions).squeeze()
                # Measures the element-wise mean squared error.
                loss = F.mse_loss(td_target, old_val)

                # log
                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                # optimize the model
                optimizer.zero_grad()
                # compute a backward pass on the loss function
                # compute the gradients of the loss function
                loss.backward()
                # take a step with the optimizer
                optimizer.step()

            # update target network
            if global_step % args.target_network_frequency == 0:
                # parameters() makes all parameters accessible
                # zip() iterate over multiple sequences return a single iterable object,
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    # update target network according update rate 
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )
   
    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.pt"   
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")
        if args.track:
            wandb.save(f"runs/{run_name}/{args.exp_name}.pt", policy="now")


    envs.close()
    writer.close()