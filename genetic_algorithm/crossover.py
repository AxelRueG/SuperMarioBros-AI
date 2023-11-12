 
import numpy as np
from typing import Tuple

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