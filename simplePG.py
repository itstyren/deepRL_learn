import torch
# building and training neural networks
import torch.nn as nn
import gymnasium as gym
# Spaces describe mathematical sets and are used in Gym to specify valid actions and observations
from gymnasium.spaces import Discrete, Box
#  parameterizable probability distributions and sampling functions
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np

def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    '''
    Build a feedforward neural network (Multi-Layer Perceptron).
    :params: the size of each layer (input+hidden+output), activation function, outut placeholder
    :returns: A sequential container, the layers in a Sequential are connected in a cascading way.
    '''
    layers = []
    for j in range(len(sizes)-1):
        # first loop is tanh, second loop is identity
        act = activation if j < len(sizes)-2 else output_activation
        # Linear(n,m) is a module that creates single layer feed forward network with n inputs and m output
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    # by using Sequential(), you can quickly implement sequential modules such that you are not required to write the forward definition
    # the layers are sequentially called on the outputs.
    # here star(*) ueses to  unpack an iterable object
    return nn.Sequential(*layers)



def train(env_name='CartPole-v1', hidden_sizes=[32], lr=1e-2, epochs=50, batch_size=5000, render=False):
    # make environment, check spaces, get obs / act dims
    env = gym.make(env_name)
    # assert obs and action
    assert isinstance(env.observation_space, Box), \
        "This example only works for envs with continuous state spaces."
    assert isinstance(env.action_space, Discrete), \
        "This example only works for envs with discrete action spaces."    
    
    obs_dim = env.observation_space.shape[0]
    # n (int) – The number of elements of this space.
    n_acts = env.action_space.n

    # make core of policy network (Multi-Layer Perceptron)
    logits_net = mlp(sizes=[obs_dim]+hidden_sizes+[n_acts])

    # make function to compute action distribution
    def get_policy(obs):
        # Pass the input through the network
        logits = logits_net(obs)
        return Categorical(logits=logits) 
    
    # make action selection function (outputs int actions, sampled from policy)
    def get_action(obs):
        # only one integer action output
        return get_policy(obs).sample().item()
    
    # make loss function whose gradient, for the right data, is policy gradient
    def compute_loss(obs, act, weights):
        logp = get_policy(obs).log_prob(act)
        return -(logp * weights).mean()
    
    # make optimizer
    optimizer = Adam(logits_net.parameters(), lr=lr)

    # for training policy
    def train_one_epoch():
        # make some empty lists for logging.
        batch_obs = []          # for observations
        batch_acts = []         # for actions
        batch_weights = []      # for R(tau) weighting in policy gradient (tau means trajectory))
        batch_rets = []         # for measuring episode returns
        batch_lens = []         # for measuring episode lengths

        # reset episode-specific variables
        obs = env.reset()[0]       # first obs comes from starting distribution
        done = False            # signal from environment that episode is over
        ep_rews = []            # list for rewards accrued throughout ep

        # render first episode of each epoch
        finished_rendering_this_epoch = False
        # collect experience by acting in the environment with current policy
        while True:

            # rendering
            if (not finished_rendering_this_epoch) and render:
                env.render()
            # save obs (shallow copy)
            batch_obs.append(obs.copy())

            # act in the environment (Converts numpy data into a tensor)
            act = get_action(torch.as_tensor(obs, dtype=torch.float32))
            # next observation due to the agent actions, reward, terminal state 
            obs, rew, done, _, info = env.step(act)
            # save action, reward
            batch_acts.append(act)
            ep_rews.append(rew)

            # done one episode
            if done:
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # the weight for each logprob(a|s) is R(tau)
                # += add elements to an existing list
                # 这个episode越长获得权重越高（times ep_len）
                batch_weights += [ep_ret] * ep_len

                # reset episode-specific variables
                obs, done, ep_rews = env.reset()[0], False, []

                # won't render again this epoch
                finished_rendering_this_epoch = True

                # end experience loop if we have enough of it
                if len(batch_obs) > batch_size:
                    break
        # take a single policy gradient update step
        # First, clear the gradient buffers
        optimizer.zero_grad()
        # compute the loss function
        batch_loss = compute_loss(obs=torch.as_tensor(np.array(batch_obs, dtype=np.float32)),
                                  act=torch.as_tensor(batch_acts, dtype=torch.int32),
                                  weights=torch.as_tensor(batch_weights, dtype=torch.float32)
                                  )
        # compute a backward pass on the loss function
        # compute the gradients of the loss function
        batch_loss.backward()
        # take a step with the optimizer
        optimizer.step()
        return batch_loss, batch_rets, batch_lens



    # training loop
    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch()
        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # define command-line interfaces argument
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v1')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-2)
    # extracted input
    args = parser.parse_args()
    print('\nUsing simplest formulation of policy gradient.\n')
    train(env_name=args.env_name, render=args.render, lr=args.lr)