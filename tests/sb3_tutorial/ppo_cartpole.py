import gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
# MlpPolicy is multi-layered-perceptron policy

env = gym.make('CartPole-v1')
model = PPO(MlpPolicy, env, verbose=0)

def evaluate(model, num_episodes=100):
    env = model.get_env()
    all_episode_rewards = []
    for i in range(num_episodes):
        episode_rewards = []
        done = False
        obs = env.reset()
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            episode_rewards.append(reward)

        all_episode_rewards.append(sum(episode_rewards))

    mean_episode_reward = np.mean(all_episode_rewards)
    print("mean:", mean_episode_reward, "num episodes:", num_episodes)

    return mean_episode_reward

mean_reward_before_train = evaluate(model, num_episodes=100)


# get video
import os
os.system
