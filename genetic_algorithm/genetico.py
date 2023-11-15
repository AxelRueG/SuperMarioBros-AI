import numpy as np
import random, os.path as op, math
from typing import List, Optional, Dict, Tuple

from config import Config

from genetic_algorithm.utils import save_mario, load_mario

from genetic_algorithm.gen_operator import tournament_selection, gaussian_mutation, simulated_binary_crossover as SBX
from genetic_algorithm.individual import Player
from genetic_algorithm.population import Population

class Genetico:
    '''
    '''
    def __init__(self, config: Optional[Config] = None):
        self.config = config
    
        self.current = 0
        self.best_fitness = 0.0
        self.generacion = 0

        individuos: List[Player] = []

        # debo generer los individos alearoteos
        for _ in range(self.config.Selection["num_offspring"]):
            individuos.append(Player(self.config))

        # o debo cargarlos de algun archivo

        self.poblacion = Population(individuos)
        self.player = self.poblacion.individuals[self.current]


    def next_generation(self) -> None:
    
        print(f'---- Fin Generacion: {self.generacion} ----------------------------------------------')
        mejor_individuo = self.poblacion.fittest_individual
        self.best_fitness = mejor_individuo.fitness
        print(f'Mejor fitness: {self.best_fitness}, Max dististancia recorrida: {mejor_individuo.farthest_x}')
        num_wins = sum(individual.did_win for individual in self.poblacion.individuals)
        tam_pob = self.poblacion.num_individuals
        print(f'Ganadores: ~{(float(num_wins)/tam_pob*100):.2f}%')

        # ---- Guardar el mejor individuo de la generacion -----------------------------------------
        if self.config.Statistics["save_best_individual_from_generation"]:
            folder = self.config.Statistics["save_best_individual_from_generation"]
            best_ind_name = 'best_ind_gen{}'.format(self.generacion)
            best_ind = self.poblacion.fittest_individual
            save_mario(folder, best_ind_name, best_ind)

        # Elitismo
        next_pop = [mejor_individuo]

        # ------------------------------------------------------------------------------------------
        #                             genero la nueva poblacion
        # ------------------------------------------------------------------------------------------
        while len(next_pop) < self.poblacion.num_individuals - 1:
            p1, p2 = tournament_selection(self.poblacion, 2, self.config.Crossover["tournament_size"])

            L = len(p1.network.layer_nodes)
            c1_params = {}
            c2_params = {}

            # Cada W_l se tratan como su propio cromosoma.
            # Debido a esto necesito realizar un cruce/mutación en cada cromosoma entre padres
            for l in range(1, L):
                p1_W_l = p1.network.params['W' + str(l)]
                p2_W_l = p2.network.params['W' + str(l)]

                # Crossover
                c1_W_l, c2_W_l = self.crossover(p1_W_l, p2_W_l)
                # Mutation
                self.mutation(c1_W_l, c2_W_l)

                # Assign children from crossover/mutation
                c1_params['W' + str(l)] = c1_W_l
                c2_params['W' + str(l)] = c2_W_l

                #  Clip to [-1, 1]
                np.clip(c1_params['W' + str(l)], -1, 1, out=c1_params['W' + str(l)])
                np.clip(c2_params['W' + str(l)], -1, 1, out=c2_params['W' + str(l)])

            # Creo los nuevos elementos
            c1 = Player(self.config, c1_params)
            c2 = Player(self.config, c2_params)

            next_pop.extend([c1, c2])

        random.shuffle(next_pop)                        # Mezclamos los individuos
        self.poblacion.individuals = next_pop           # Actualizamos poblacion

        # Aumentamos la generacion y setteamos el primer jugador
        self.generacion += 1
        self.current = 0
        self.player = self.poblacion.individuals[self.current]


    def crossover(self, parent1_weights: np.ndarray, parent2_weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        eta = self.config.Crossover["sbx_eta"]
        # SBX: pesos
        child1_weights, child2_weights = SBX(parent1_weights, parent2_weights, eta)
        return child1_weights, child2_weights


    def mutation(self, child1_weights: np.ndarray, child2_weights: np.ndarray) -> None:
        mutation_rate = self.config.Mutation["mutation_rate"]
        scale = self.config.Mutation["gaussian_mutation_scale"]

        if self.config.Mutation["mutation_rate_type"] == 'dynamic':
            mutation_rate = mutation_rate / math.sqrt(self.generacion + 1)
        
        # Mutate weights
        gaussian_mutation(child1_weights, mutation_rate, scale=scale)
        gaussian_mutation(child2_weights, mutation_rate, scale=scale)

    def next_individuo(self):
        print(f'individuo actual {self.current}')
        self.current += 1
        # si todavia no termine de recorrer todos los individuos
        if (self.current < self.poblacion.num_individuals):
            self.player = self.poblacion.individuals[self.current]
        else:
            self.next_generation()
