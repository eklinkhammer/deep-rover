from random import shuffle

import numpy as np
import math

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

    def set_population(self, population):
        """ Initializes with an existing population of FeedForwardNeuralNets """
        self._population = population

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
        
        pools = np.array(teams).T.tolist()
        scores_pools = np.array(scores).T.tolist()
        self.tournamentN(pools, scores_pools, 4, 0.4)

        return scores

    def best_team(self):
        teams = np.array(self._population).T.tolist()
        scores = [self.fitness_evaluator.fitness(team) for team in teams]

        team_scores = [sum(s) for s in scores]
        max_score_i = np.argmax(team_scores)
        return sum(team_scores) / len(team_scores), team_scores[max_score_i], teams[max_score_i]
    
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

    def tournamentN(self, pop, scores, tournament_size, survivor_rate):
        """ Tournament selection and breeding function.
        """

        #print (["%.2f" % x for x in scores])
        survivor_count = math.floor(self._pool_size * survivor_rate)
        if survivor_count < 1:
            survivor_count = 1
        for pool_i in range(self._num_pools):
            winning_nets = {}
            for net_i in range(survivor_count):
                rand_indices = [math.floor(min(self._pool_size, x)) for x in np.random.rand(tournament_size) * self._pool_size]

                # print (scores)
                # print (rand_indices)
                # print (scores[pool_i])
                scores_rand_indices = [scores[pool_i][i] for i in rand_indices]
                
                winning_index = np.argmax(scores_rand_indices)
                winning_net = pop[pool_i][rand_indices[winning_index]]
                if winning_net in winning_nets:
                    pass
                else:
                    winning_nets[winning_net] = 0
                    self._population[pool_i][net_i].copy_weights(winning_net)

            for new_nets in range(survivor_count + 1, self._pool_size):
                parent_index = math.floor(np.random.rand(1) * survivor_count)
                self._population[pool_i][new_nets].copy_weights(self._population[pool_i][parent_index])
                self._population[pool_i][new_nets].mutate()
                




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

    def merge_pops(self, pop1, pop2):
        [a.extend(b) for (a,b) in zip(pop1, pop2)]
        return pop1

    def noise(self, pop):
        for pool in pop:
            for net in pool:
                net.mutate()

    def copy_population(self, population):
        new_pop = []
        for pool in population:
            new_pool = []
            for net in pool:
                new_pool.append(net.deep_copy())
            new_pop.append(new_pool)
        return new_pop
                
    def elite_noise(self):
        # Copy weights from first half
        for pool in self._population:
            half_p = len(pool) // 2
            for i in range(half_p):
                # pool[i + half_p].copy_weights(pool[i])
                pool[i + half_p].mutate()
