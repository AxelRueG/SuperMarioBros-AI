import numpy as np
from typing import List
from genetic_algorithm.player import Player

class Population(object):
    #crea una lista de individuos
    def __init__(self, individuals: List[Player]):
        self.individuals = individuals
    
    #devuelve la cantida de individuos
    @property
    def num_individuals(self) -> int:
        return len(self.individuals)
    
    @property
    def num_genes(self) -> int:
        return self.individuals[0].chromosome.shape[1]

    @property
    def average_fitness(self) -> float:
        return (sum(individual.fitness for individual in self.individuals) / float(self.num_individuals))

    @property
    def fittest_individual(self) -> Player:
        return max(self.individuals, key = lambda individual: individual.fitness)

    def calculate_fitness(self) -> None:
        for individual in self.individuals:
            individual.calculate_fitness()

    def get_fitness_std(self) -> float:
       return np.std(np.array([individual.fitness for individual in self.individuals]))
