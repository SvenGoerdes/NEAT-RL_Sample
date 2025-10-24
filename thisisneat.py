# pip3 install gym
# pip3 install neat-python

# for gym stuff: 
# apt install xvfb ffmpeg xorg-dev libsdl2-dev swig cmake
# pip3 install gym[box2d]

import multiprocessing
import os
import pickle

import neat
import numpy as np
#import cart_pole
import gymnasium as gym



runs_per_net = 2



# Use the NN network phenotype and the discrete actuator force function.
def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    fitnesses = []

    for runs in range(runs_per_net):
        env = gym.make("Acrobot-v1")

        observation, _ = env.reset()
        fitness = 0.0
        terminated = False
        truncated = False
        while not terminated and not truncated:
            # Network output -> discrete action for Acrobot (0, 1, or 2)
            output = net.activate(observation)
            if len(output) == 3:
                action = int(np.argmax(output))
            elif len(output) == 1:
                # Map single output in [-inf, inf] to 3 bins {0,1,2}
                x = output[0]
                action = int(np.digitize([x], [-0.33, 0.33])[0])
            else:
                # Fallback: use argmax over whatever outputs exist
                action = int(np.argmax(output))

            observation, reward, terminated, truncated, info = env.step(action)
            fitness += reward

        fitnesses.append(fitness)

    return np.mean(fitnesses)


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)


def run():
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    winner = pop.run(pe.evaluate)

    # Save the winner.
    with open('winner', 'wb') as f:
        pickle.dump(winner, f)

    print(winner)




if __name__ == '__main__':
    run()
