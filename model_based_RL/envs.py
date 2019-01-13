import sys
import os

cwd = os.path.dirname(os.path.realpath(__file__))
env_path = os.path.join(cwd, '..', 'rl-generalization')
env_path2 = os.path.join(env_path, 'sunblaze_envs')

sys.path.append(env_path)
sys.path.append(env_path2)
import gym
from sunblaze_envs.classic_control import *
from sunblaze_envs import *


class CartPoleEnv:
    def __init__(self, force_mag=10, length=2, mass=2):
        self.force_mag = force_mag
        self.length = length
        self.mass = mass
        self.env = PredefinedCartPoleEnv(force_mag, length, mass)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self):
        self.env = PredefinedCartPoleEnv(self.force_mag, self.length, self.mass)
        obs = self.env.reset()
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info
