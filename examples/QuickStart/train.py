#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import gym
import numpy as np
import parl
from parl.utils import logger
from parl.env import CompatWrapper, is_gym_version_ge
from cartpole_model import CartpoleModel
from cartpole_agent import CartpoleAgent
import argparse

LEARNING_RATE = 1e-3


# train an episode
def run_train_episode(agent, env):
    obs_list, action_list, reward_list = [], [], []
    obs = env.reset()
    while True:
        obs_list.append(obs)
        action = agent.sample(obs)
        action_list.append(action)

        obs, reward, done, info = env.step(action)
        reward_list.append(reward)

        if done:
            break
    return obs_list, action_list, reward_list


# evaluate 5 episodes
def run_evaluate_episodes(agent, eval_episodes=5, render=False):
    # Compatible for different versions of gym
    if is_gym_version_ge("0.26.0") and render:  # if gym version >= 0.26.0
        env = gym.make('CartPole-v1', render_mode="human")
    else:
        env = gym.make('CartPole-v1')
    env = CompatWrapper(env)

    eval_reward = []
    for i in range(eval_episodes):
        obs = env.reset()
        episode_reward = 0
        while True:
            action = agent.predict(obs)
            obs, reward, isOver, _ = env.step(action)
            episode_reward += reward
            if render:
                env.render()
            if isOver:
                break
        eval_reward.append(episode_reward)
    return np.mea