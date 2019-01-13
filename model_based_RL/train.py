import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import numpy as np
from model import pred_net
from buffer_utils import ReplayBuffer
from envs import CartPoleEnv


def sample_action(net, obs, args):
    raise NotImplementedError


def train_model(net, replay_buffer, optimizer, args):
    batch_size = args.batch_size
    sample_strategy = args.strategy
    criterion = nn.MSELoss()
    samples = replay_buffer.sample(batch_size, sample_strategy)
    latent = torch.from_numpy(np.array([args.force_mag, args.length, args.mass]))
    total_step = samples.size()[1]

    init_state = samples[:, 0].state
    action_seq = [samples[:, i].action for i in range(total_step - 1)]
    reward = [samples[:, i].reward for i in range(1, total_step)]
    state = [samples[:, i].state for i in range(1, total_step)]

    if torch.cuda.is_available():
        latent = latent.cuda()
        init_state = init_state.cuda()
        action_seq = action_seq.cuda()
        reward = reward.cuda()
        state = state.cuda()

    pred_state, pred_reward = net(init_state, action_seq, latent)
    loss = criterion(state, pred_reward) + args.reward_factor * criterion(reward, pred_reward)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def train(args, env, total_steps=40000000):

    # initialize models and buffers
    obs = env.reset()
    s_dim = obs.shape[0]
    a_dim = env.action_space.n
    p_dim = 3
    batch_size = args.batch_size
    train_net = pred_net(s_dim, a_dim, p_dim)
    if torch.cuda.is_available():
        train_net = train_net.cuda()
    replay_buffer = ReplayBuffer(args)
    optimizer = optim.Adam(train_net.parameters(), lr=args.lr, amsgrad=True)

    for i in range(total_steps):
        obs = replay_buffer.encode_recent_observations()
        action = sample_action(train_net, obs, args)
        obs, reward, done, info = env.step(action)
        replay_buffer.store(obs, reward, done, info)

        if i % args.train_freq == 0:
            train_model(train_net, replay_buffer, optimizer, batch_size)

        if done:
            obs = env.reset()
            obs = replay_buffer.encode_recent_observations()


def main(args):
    env = CartPoleEnv(args.force_mag, args.length, args.mass)
    train(args, env)
