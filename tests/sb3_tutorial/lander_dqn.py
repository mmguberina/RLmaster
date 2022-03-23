import gym
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make('CartPole-v1')
model = A2C('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=int(1e5))
#model.save("dqn_cartpole2")
#del model

model = A2C.load("dqn_cartpole2", env=env)
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    env.render()
    if dones:
        env.reset()
