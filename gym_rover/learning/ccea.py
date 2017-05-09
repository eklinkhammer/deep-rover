from random import shuffle

import numpy as np

from gym_rover.learning.ffnet import FeedForwardNeuralNet

class CCEA(object):
    """ Cooperative Coevolutionary Algorithm """
    
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
            breeding = self.elite_noise
        self.breeding_function = breeding

        if selection is None:
            selection = self.tournament2
        self.selection_function = selection

    def generation(self, debug=False):
        """ Evolves the population of neural networks one generation.
            In the evolution process, the intial population is mutated. The 
                mutated population is assessed for fitness, and the population 
                is culled according to both fitness and a selection strategy.
        """

        # Mutate population
        # self.breeding_function()

        # Form teams
        for pool in self._population:
            shuffle(pool)
        teams = np.array(self._population).T.tolist()

        # Assign Fitness
        scores = [self.fitness_evaluator.fitness(team, debug) for team in teams]
        if debug:
            print ('Scores: ')
            print (scores)
        # Select winners
        self._population = self.selection_function(teams, scores)
        if debug:
            print ('Post-tournament scores: ')
            scores2 = [self.fitness_evaluator.fitness(team, debug) for team in np.array(self._population).T.tolist()]
            print (scores2)
        # Breed after selection. Selection is inplace, second half of list should be
        # discarded.
        self.breeding_function()
        if debug:
            print ('Post-breeding scores: ')
            scores2 = [self.fitness_evaluator.fitness(team, debug) for team in np.array(self._population).T.tolist()]
            print (scores2)
        scores.sort()
        return scores[-3:]

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

    def copy_population(self, population):
        new_pop = []
        for pool in population:
            new_pool = []
            for net in pool:
                new_pool.append(net.deep_copy())
            new_pop.append(new_pool)
        return new_pop

    def merge_pops(self, pop1, pop2):
        [a.extend(b) for (a,b) in zip(pop1, pop2)]
        return pop1

    def noise(self, pop):
        for pool in pop:
            for net in pool:
                net.mutate()
                
    def elite_noise(self):
        # Copy weights from first half
        for pool in self._population:
            half_p = len(pool) // 2
            for i in range(half_p):
                # pool[i + half_p].copy_weights(pool[i])
                pool[i + half_p].mutate()

        # # Mutate second half
        # copies = self.copy_population(self._population)
        # self.noise(copies)
        # self.merge_pops(self._population, copies)

    def tournament2(self, teams, scores):
        ''' Time for some inplace chicanery '''

        pop = np.array(teams).T.tolist()
        scores_pop = np.array(scores).T.tolist()

        for pool_index in range(len(pop)):
            pool = pop[pool_index]
            pool_score = scores_pop[pool_index]

            half_p = len(pool) // 2
            for net_index in range(half_p):
                score_1 = pool_score[net_index]
                score_2 = pool_score[net_index + half_p]
                if score_2 > score_1:
                    pop[pool_index][net_index].copy_weights(pop[pool_index][net_index + half_p])
                else:
                    pop[pool_index][net_index + half_p].copy_weights(pop[pool_index][net_index])
        return pop
