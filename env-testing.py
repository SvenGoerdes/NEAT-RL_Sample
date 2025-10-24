import gymnasium as gym 


#env = gym.make("BipedalWalker-v2")
env = gym.make("BipedalWalker-v3", render_mode="human")

observation = env.reset()

print(observation)
print(env.action_space)

terminated = False
truncated = False
while not terminated and not truncated:
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
    print(env.action_space.sample())

    env.render()
    