import numpy as np
import random
from typing import List
from .population import Population

from mario import Mario

# Se definen los metodos de seleccion para la siguiente generacion de individuos
def elitism_selection(population: Population, num_individuals: int) -> List[Mario]:
    """
    Se retorna los individuos elegidos a partir de ordenar la lista de la poblacion de forma descendente
    a partir de su puntaje de fitness
    """
    individuals = sorted(population.individuals, key = lambda individual: individual.fitness, reverse=True)
    return individuals[:num_individuals]

def roulette_wheel_selection(population: Population, num_individuals: int) -> List[Mario]:
    """
    Se retorna los individuos elegidos usando el metodo de la ruleta. Se arma la ruleta asignando
    porciones segun el fitness. Se elige un valor aleatorio y se hace la eleccion de los individuos
    """
    selection = []
    wheel = sum(individual.fitness for individual in population.individuals)
    for _ in range(num_individuals):
        pick = random.uniform(0, wheel)
        current = 0
        for individual in population.individuals:
            current += individual.fitness
            if current > pick:
                selection.append(individual)
                break

    return selection

def tournament_selection(population: Population, num_individuals: int, tournament_size: int) -> List[Mario]:
    """
    Se generan torneos de tamaño aleatorio para seleccionar cada individuo donde va quedando el ganador de
    cada uno y pasa a la lista de seleccionados.
    """
    selection = []
    for _ in range(num_individuals):
        tournament = np.random.choice(population.individuals, tournament_size)
        best_from_tournament = max(tournament, key = lambda individual: individual.fitness)
        selection.append(best_from_tournament)

    return selection