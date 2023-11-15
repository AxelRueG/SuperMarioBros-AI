import numpy as np
import random
from typing import List, Tuple, Union, Optional
from .population import Population

from genetic_algorithm.utils import Player


# --------------------------------------------------------------------------------------------------
#                   OPERADORES DE SELECCION
# --------------------------------------------------------------------------------------------------
# Se definen los metodos de seleccion para la siguiente generacion de individuos
def elitism_selection(population: Population, num_individuals: int) -> List[Player]:
    """
    Se retorna los individuos elegidos a partir de ordenar la lista de la poblacion de forma descendente
    a partir de su puntaje de fitness
    """
    individuals = sorted(population.individuals, key = lambda individual: individual.fitness, reverse=True)
    return individuals[:num_individuals]

def roulette_wheel_selection(population: Population, num_individuals: int) -> List[Player]:
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

def tournament_selection(population: Population, num_individuals: int, tournament_size: int) -> List[Player]:
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


# --------------------------------------------------------------------------------------------------
#                   OPERADORES DE CRUZA
# --------------------------------------------------------------------------------------------------
# Se definen diferentes metodos para realizar las cruzas de los individuos
def simulated_binary_crossover(parent1: np.ndarray, parent2: np.ndarray, eta: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Define un metodo de cruza orientado a representaciones de punto flotante, simulando el metodo de
    cruza one-point para representaciones binarias
    Para grandes valores de eta, hay mayor probabilidad de que el hijo sea creado mas parecido a los padres
    Para pequeños valores de eta, hay mayor probabilidad de que el hijo sea creado menos parecido a los padres
    Ecuacion 9.9, 9.10, 9.11
    """    
    # Calculo de Gamma (Ec. 9.11)
    rand = np.random.random(parent1.shape)
    gamma = np.empty(parent1.shape)
    gamma[rand <= 0.5] = (2 * rand[rand <= 0.5]) ** (1.0 / (eta + 1))  # Primer caso de la ecuacion 9.11
    gamma[rand > 0.5] = (1.0 / (2.0 * (1.0 - rand[rand > 0.5]))) ** (1.0 / (eta + 1))  # Segundo caso de la ec

    # Calculo del cromosoma hijo 1 (Ec. 9.9)
    chromosome1 = 0.5 * ((1 + gamma)*parent1 + (1 - gamma)*parent2)
    # Calculo del cromosoma hijo 2 (Ec. 9.10)
    chromosome2 = 0.5 * ((1 - gamma)*parent1 + (1 + gamma)*parent2)

    return chromosome1, chromosome2

def uniform_binary_crossover(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Define el metodo de cruza donde se crea una mascara con 0 y 1.
    Dependiendo como se defina la funcion, al recorrer los genes de los hijos y la mascara, se realiza un swap
    o no entre los genes de los hijos dependiendo el valor de la mascara
    """
    offspring1 = parent1.copy()
    offspring2 = parent2.copy()
    
    mask = np.random.uniform(0, 1, size=offspring1.shape) # Mascara con 0 y 1, del tamaño del individuo
    offspring1[mask > 0.5] = parent2[mask > 0.5] # Swap del gen
    offspring2[mask > 0.5] = parent1[mask > 0.5] # Swap del gen

    return offspring1, offspring2

def single_point_binary_crossover(parent1: np.ndarray, parent2: np.ndarray, major='r') -> Tuple[np.ndarray, np.ndarray]:
    """
    Define el metodo de cruza one-point donde se elige un punto del cromosoma de los padres y se hace un swap de
    los bits para crear a los hijos. Aca hace algo mas complicado y no lo entiendo
    """
    offspring1 = parent1.copy()
    offspring2 = parent2.copy()

    rows, cols = parent2.shape
    row = np.random.randint(0, rows)
    col = np.random.randint(0, cols)

    if major.lower() == 'r':
        offspring1[:row, :] = parent2[:row, :]
        offspring2[:row, :] = parent1[:row, :]

        offspring1[row, :col+1] = parent2[row, :col+1]
        offspring2[row, :col+1] = parent1[row, :col+1]
    elif major.lower() == 'c':
        offspring1[:, :col] = parent2[:, :col]
        offspring2[:, :col] = parent1[:, :col]

        offspring1[:row+1, col] = parent2[:row+1, col]
        offspring2[:row+1, col] = parent1[:row+1, col]

    return offspring1, offspring2


# --------------------------------------------------------------------------------------------------
#                   OPERADORES DE MUTACION
# --------------------------------------------------------------------------------------------------
# Se definen diferentes metodos para realizar la mutacion de los individuos
def gaussian_mutation(chromosome: np.ndarray, prob_mutation: float, 
                      mu: List[float] = None, sigma: List[float] = None,
                      scale: Optional[float] = None) -> None:
    """
    Se realiza la mutacion gaussiana para cada gen en el individuo a partir de la probabilidad.
    Si mu y sigma estan definidos entonces la distribucion gaussiana se define a partir de esos valores,
    en caso contrario se obtienen de N(0, 1) a partir del tamaño del individuo
    """
    # Determina que genes seran los reciban una mutacion
    mutation_array = np.random.random(chromosome.shape) < prob_mutation
    # Si mu y sigma estan definidos, se crea la distribucion gaussiana para cada uno
    if mu and sigma:
        gaussian_mutation = np.random.normal(mu, sigma)
    # En caso contrario, se centra alrededor de N(0,1)
    else:
        gaussian_mutation = np.random.normal(size=chromosome.shape)
    
    if scale:
        gaussian_mutation[mutation_array] *= scale

    # Actualizacion del cromosoma
    chromosome[mutation_array] += gaussian_mutation[mutation_array]

def random_uniform_mutation(chromosome: np.ndarray, prob_mutation: float,
                            low: Union[List[float], float],
                            high: Union[List[float], float]
                            ) -> None:
    """
    Se muta de forma aleatoria cada gen de un individuo a partir de la probabilidad.
    Si un gen es seleccionado, se le asignara un valor con una probabilidad uniforme entre [low,high)
    [low, high) esta definido para cada gen para abarcar todo el rango posible de valores
    """
    assert type(low) == type(high), 'low and high deben ser del mismo tipo'
    mutation_array = np.random.random(chromosome.shape) < prob_mutation
    if isinstance(low, list):
        uniform_mutation = np.random.uniform(low, high)
    else:
        uniform_mutation = np.random.uniform(low, high, size=chromosome.shape)
    chromosome[mutation_array] = uniform_mutation[mutation_array]

def uniform_mutation_with_respect_to_best_individual(chromosome: np.ndarray, best_chromosome: np.ndarray, prob_mutation: float) -> None:
    """
    Se muta de forma aleatoria cada gen de un individuo a partir de la probabilidad.
    Si el gen es elegido para mutar, se busca acercar su valor al valor del mejor cromosoma (individuo).
    """
    mutation_array = np.random.random(chromosome.shape) < prob_mutation
    uniform_mutation = np.random.uniform(size=chromosome.shape)
    chromosome[mutation_array] += uniform_mutation[mutation_array] * (best_chromosome[mutation_array] - chromosome[mutation_array])