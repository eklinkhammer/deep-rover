from random import shuffle

import numpy as np

from gym_rover.learning.ffnet import FeedForwardNeuralNet

class CCEA(object):
    """ Cooperative Coevolutionary Algorithm """


    ELITIST_NOISE = elite_noise
    TOURNAMENT = tournament2
    
    def __init__(self, pools, pool_size, layers, active_funcs, mutation_rate,
                 fitness, breeding=None, selection=None):
        """ Create CCEA Object. Manages the evolution of a population of
                feed forward neural networks.

        Args:
            pools (int): The number of pools in the population. Pools are
                           roughly each agent's population.
            pool_size (int): The number of networks per pool (and per agent)
            layers ([int]): The number of nodes per layer in network.
            active_funcs ([string]): The activation functions per layer.
            mutation_rate (double): The percent of weights mutated per mutation.
            fitness (object): Has fitness method that takes team of neural nets
                                  and returns a score for the team.
            breeding (string): Specify breeding function.
            selection (string): Specify selection function

        """
        self._num_pools = pools
        self._pool_size = pool_size

        self._create_population(layers, active_funcs, mutation_rate)

        self.fitness_evaluator = fitness
        
        if breeding is None:
            breeding = self.ELITIST_NOISE
        self.breeding_function = breeding

        if selection is None:
            selection = self.TOURNAMENT
        self.selection_function = selection

    def generation(self):
        """ Evolves the population of neural networks one generation.
            In the evolution process, the intial population is mutated. The 
                mutated population is assessed for fitness, and the population 
                is culled according to both fitness and a selection strategy.
        """

        # Mutate population
        self._population = self.breeding_function(self._population)


        # Form teams
        for pool in self._population:
            pool.shuffle()
        teams = np.array(self._population).T.tolist()

        # Assign Fitness
        scores = [self.fitness_evaluator.fitness(team) for team in teams]

        # Select winners
        self._population = self.selection_function(teams, scores)

    def best_team(self):
        teams = np.array(self._population).T.tolist()
        scores = [self.fitness_evaluator.fitness(team) for team in teams]

        team_scores = [sum(s) for s in scores]

        teams_scores = list(zip(team_scores, teams))

        teams_scores.sort(key=lambda x : x[0])

        return teams_scores[0][1]
    
    def _create_population(self, layers, actives, rate):
        """ Create a population of neural networks. Populations will be 
                subdivided into pools of (initial) determined size.
        
        Mutates:
            self: Adds a _population attribute that will be a 2D list of ff nets
        """
        self._population = []
        for i in range(self._num_pools):
            self._population.append([])
            for j in range(self._pool_size):
                self._population[i].append(FeedForwardNeuralNet(layers, actives,
                                                                rate))
        
    def elite_noise(self):
        for pool in self._population:
            for net in pool:
                pool.append(net.deep_copy().mutate())

    def tournament2(self, teams, scores):
        pop = np.array(teams).T.tolist()
        scores_pop = np.array(scores).T.tolist()

        new_pop = []
        for pool_index in range(len(pop)):
            new_pop.append([])
            pool = pop[pool_index]
            pool_score = scores_pop[pool_index]

            for net_index in range(len(pool) // 2):
                score_1 = pool_score[net_index]
                score_2 = pool_score[net_index + len(pool) // 2]
                if score_1 > score_2:
                    winner = pool[net_index]
                else:
                    winner = pool[net_index + len(pool) // 2]

                new_pop[pool_index].append(winner)

        return new_pop
