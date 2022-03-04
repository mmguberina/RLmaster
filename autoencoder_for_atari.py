import gym
import matplotlib.pyplot as plt

env = gym.make('PongNoFrameskip-v4')
env.reset()
obs, reward, done, info = env.step(0)

plt.imshow(obs)
plt.savefig('img.png')
