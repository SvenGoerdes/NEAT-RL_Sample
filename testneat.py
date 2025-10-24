import os
import pickle
import neat
import gymnasium as gym
import numpy as np

# load the winner
with open('winner', 'rb') as f:
    c = pickle.load(f)

print('Loaded genome:')
print(c)

# Load the config file, which is assumed to live in
# the same directory as this script.
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config')
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)

net = neat.nn.FeedForwardNetwork.create(c, config)


env = gym.make("Acrobot-v1", render_mode="human")
observation, info = env.reset()

terminated = False
truncated = False
while not terminated and not truncated:
    output = net.activate(observation)
    if len(output) == 3:
        action = int(np.argmax(output))
    elif len(output) == 1:
        x = output[0]
        action = int(np.digitize([x], [-0.33, 0.33])[0])
    else:
        action = int(np.argmax(output))

    observation, reward, terminated, truncated, info = env.step(action)
    env.render()

