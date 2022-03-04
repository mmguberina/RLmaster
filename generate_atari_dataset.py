import gym
import numpy as np
import torch
import cv2
from PIL import Image
import imagehash

env = gym.make('PongNoFrameskip-v4')
env.reset()
#env.render()

#np.random.seed(args.seed)
#torch.manual_seed(args.seed)
#train_envs.seed(args.seed)
#test_envs.seed(args.seed)

n_images = 100000
images = set()

i = 0
for i in range(n_images):
    obs, reward, done, info = env.step(env.action_space.sample())
    hashed_im = imagehash.dhash(Image.fromarray(obs))
    if hashed_im not in images:
        cv2.imwrite("pong_dataset/" + str(i) + ".png", obs)
    images.add(hashed_im)
    i += 1
    if done:
        env.reset()
#    env.render()
#plt.imshow(obs)
