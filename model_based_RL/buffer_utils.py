import os
import torch
import numpy as np
from torch.autograd import Variable


class ReplayBuffer(object):
    def __init__(self, args):
        self.args = args
        self.buffer_number = 0
        self.next_index = 0

        self.episode_lens = []
        self.expert = None
        self.obs = []
        self.action = []
        self.done = []
        self.reward = []

    def get_bar(self):
        index = int(len(self.episode_lens) * self.args.expert_ratio)
        bar = max(sorted(self.episode_lens, reverse=True)[index], self.args.bar_min)
        return bar

    def can_sample_guide(self, batch_size):
        if len(self.episode_lens) == 0:
            print('empty episode_lens')
            return False
        print("Getting bar from %s" % str(self.episode_lens))
        bar = self.get_bar()
        print("Bar is %d" % bar)
        bar_index = np.where(self.expert[:self.buffer_number] >= bar)[0]
        print("Getting bar index: %s" % str(bar_index))
        return len(bar_index) >= batch_size

    def sample_guide(self, batch_size):
        raise NotImplementedError

    def sample_random(self, batch_size):
        indexes = np.random.choice(range(self.buffer_number), batch_size)
        obs_np = np.concatenate([self.obs[index][np.newaxis, :] for index in indexes], axis=0)
        obs = torch.from_numpy(obs_np.float(), requires_grad=False)  # norm?
        obs = Variable(obs)
        if torch.cuda.is_available():
            obs = obs.cuda()
        return obs

    def sample_random_continuous(self, batch_size):
        raise NotImplementedError

    def sample(self, batch_size, strategy):
        if strategy == 'sample_random':
            return self.sample_random(batch_size)
        elif strategy == 'sample_random_continuous':
            return self.sample_random_continuous(batch_size)
        else:
            raise NotImplementedError("not implemented sample strategy: {}".format(strategy))

    def _encode_observation(self, index):
        end_index = index + 1
        start_index = end_index - self.args.history_len
        if len(self.obs.shape) == 2:  # low dimensional observation
            return self.obs[end_index - 1]
        if start_index < 0 and self.buffer_number != self.args.buffer_size:
            start_index = 0
        for index in range(start_index, end_index - 1):
            if self.done[index % self.args.buffer_size]:
                start_index = index + 1
        missing_context = self.args.history_len - (end_index - start_index)

        if start_index < 0 or missing_context > 0:
            frames = [np.zeros_like(self.obs[0]) for i in range(missing_context)]
            for index in range(start_index, end_index):
                frames.append(self.obs[index % self.args.buffer_size])
            return np.concatenate(frames, axis=0)
        else:
            raise NotImplementedError

    def encode_recent_observation(self):
        assert self.buffer_number > 0
        return self._encode_observation((self.next_index - 1) % self.args.buffer_size)

    def load(self, path):
        assert(os.path.isdir(path))
        assert(os.path.exists(os.path.join(path, 'obs.npy')))
        assert(os.path.exists(os.path.join(path, 'action.npy')))
        assert(os.path.exists(os.path.join(path, 'done.npy')))
        self.obs = np.load(os.path.join(path, 'obs.npy'))
        self.action = np.load(os.path.join(path, 'action.npy'))
        self.done = np.load(os.path.join(path, 'done.npy'))

    def store(self, obs, reward, done, info):
        self.obs.append(obs)
        self.reward.append(reward)
        self.don.append(done)
        self.info.append(info)
        self.buffer_number += 1
