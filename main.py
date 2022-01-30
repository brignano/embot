import multiprocessing

import cv2
import neat
import numpy as np
import retro


class Worker:
    def __init__(self, genome, config):
        self.genome = genome
        self.config = config

    def work(self):
        self.env = retro.make('SuperMarioBros-Nes')
        self.env.reset()

        ob, _, _, _, = self.env.step(self.env.action_space.sample())

        inx = int(ob.shape[0] / 8)
        iny = int(ob.shape[1] / 8)
        done = False

        net = neat.nn.FeedForwardNetwork.create(self.genome, self.config)

        fitness = 0
        xpos = 0
        xpos_max = 0
        imgarray = []
        counter = 0

        while not done:
            ob = cv2.resize(ob, (inx, iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            ob = np.reshape(ob, (inx, iny))

            imgarray = np.ndarray.flatten(ob)

            actions = net.activate(imgarray)

            ob, rew, done, info = self.env.step(actions)

            xpos = info['xscrollLo']

            if xpos > xpos_max:
                xpos_max = xpos
                counter = 0
                fitness += 1
            else:
                counter += 1

            if counter > 250:
                done = True

        print(fitness)
        return fitness


def eval_genomes(genome, config):
    worker = Worker(genome, config)
    worker.work()


def main():
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         'config-feedforward')

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(10))

    pe = neat.ParallelEvaluator(multiprocessing.cpu_count() - 2, eval_genomes)

    winner = p.run(pe.evaluate)
    return winner


if __name__ == '__main__':
    main()
