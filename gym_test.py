# -*- coding: utf-8 -*-
"""
Created on Thu May  4 13:38:21 2017
The simplest example of gym environment implement
@author: minty
"""
import gym
from gym import wrappers

class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    outdir = '/tmp/random-agent-results'
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    agent = RandomAgent(env.action_space)

    episode_count = 100
    reward = 0
    Terminal = False

    for i in range(episode_count):
        ob = env.reset()
        while True:
            action = agent.act(ob, reward, Terminal)
            ob, reward, Terminal, _ = env.step(action)
            if Terminal:
                break
    env.close()
