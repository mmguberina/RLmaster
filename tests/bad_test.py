import gym
import tianshou as ts

env = gym.make('CartPole-v0')

train_envs = ts.env.DummyVectorEnv([lambda: gym.make('CartPole-v0') for _ in range(10)])
test_envs = ts.env.DummyVectorEnv([lambda: gym.make('CartPole-v0') for _ in range(100)])

# you want to run multiple sims
# to ensure that they are properly instantiated and
# that the random number generators are seeded properly,
# use some of the environment wrappers

import torch, numpy as np
from torch import nn

class Net(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        self.model = nn.Sequential(
                nn.Linear(np.prod(state_shape), 128), nn.ReLU(inplace=True),
                nn.Linear(128, 128), nn.ReLU(inplace=True),
                nn.Linear(128, 128), nn.ReLU(inplace=True),
                nn.Linear(128, np.prod(action_shape)),
        )

        def forward(self, obs, state=None, info={}):
            if not isinstance(obs, dtype=torch.float):
                obs = torch.tensor(obs, dtype=torch.float)
            batch = obs.shape[0]
            logits = self.model(obs.view(batch, -1))
            return logits, state

state_shape = env.observation_space.shape or env.observation_space.n
action_shape = env.action_space.shape or env.action_space.n
net = Net(state_shape, action_shape)
optim = torch.optim.Adam(net.parameters(), lr=1e-3)

policy = ts.policy.DQNPolicy(net, optim, discount_factor=0.9, estimation_step=3, target_update_freq=320)
train_collector = ts.data.Collector(policy, train_envs, ts.data.VectorReplayBuffer(20000, 10), exploration_noise=True)
test_collector = ts.data.Collector(policy, test_envs, exploration_noise=True)

ts.trainer.offline_trainer
result = ts.trainer.offline_trainer(
        policy, train_collector, test_collector,
        max_epoch=10, update_per_epoch=10000, 
        episode_per_test=100, batch_size=64,
        train_fn=lambda epoch, env_step: policy.set_eps(0.1),
        test_fn=lambda epoch, env_step: policy.set_eps(0.05),
        stop_fn=lambda mean_rewards: mean_rewards >= env.spec.reward_threshold)
print(f'Finished training! Use {result["duration"]}')

from torch.utils.tensorboard import SummaryWritter
from tianshou.utils import TensorboardLogger
writter = SummaryWritter('log/dqn')
logger = TensorboardLogger(writer)

torch.save(policy.state_dict(), 'dqn.pth')
#policy.load_state_dict(torch.load('dqn.pth'))