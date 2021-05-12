import gym
from gym import wrappers

env = gym.make('CartPole-v0')
env = wrappers.Monitor(env, './video', force=True)

state = env.reset()

for step in range(200):
    env.render()
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)

    if done:
        break
